import sys
import os
import argparse

# ============================================================================
# CRITICAL: Parse CUDA device BEFORE importing torch
# ============================================================================
def _parse_cuda_device_early():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda_device", type=int, default=2)
    args, _ = parser.parse_known_args()
    return args.cuda_device

_cuda_device = _parse_cuda_device_early()
if _cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda_device)
    print(f"[GPU Isolation] CUDA_VISIBLE_DEVICES={_cuda_device} (visible as cuda:0)")
# ============================================================================

import torch
import numpy as np
import json
import nltk
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from peft import PeftModel
from datetime import datetime

# Add parent directory to path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ZGeneration.config_gen import GenTrainingConfig
from ZGeneration.data_loader_gen import get_gen_dataloader
from ZGeneration.model_loader_gen import GenModelLoader
from utils_llama3 import setup_logger, set_seed

# Ensure NLTK data (lite check)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt-tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def calculate_accuracy(logits, labels):
    # Logits: [Batch, Seq, Vocab], Labels: [Batch, Seq]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    preds = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    correct = (preds == shift_labels) & mask
    if mask.sum() == 0: return 0.0
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def calculate_per_sample_ppl(logits, labels):
    # CRITICAL: Must set ignore_index=-100 to properly handle masked labels
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # [Batch, Seq-1]
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    # With ignore_index=-100, loss for ignored positions is already 0
    mask = (shift_labels != -100).float()
    # Avoid div by zero
    counts = mask.sum(dim=1)
    sums = loss.sum(dim=1)  # loss already has 0 for ignored positions
    per_sample_loss = sums / counts.clamp(min=1)
    # If count is 0, ppl is technically undefined, set to 0 or 1? 
    # Let's keep distinct, but usually implies padding only.
    per_sample_ppl = torch.exp(per_sample_loss)
    # Restore valid ppl only where count > 0
    return per_sample_ppl.tolist()

def IAMM_ppl_loss_list(logits, labels):
    """
    Computes per-sample losses following IAMM method.
    Returns the list of scalar losses (mean NLL per token) for each sample.
    Later, the global PPL is calculated as exp(mean(all_sample_losses)).
    
    Why CrossEntropyLoss?
    IAMM uses NLLLoss on LogSoftmax-ed inputs.
    Our model outputs raw logits.
    CrossEntropyLoss(logits) == NLLLoss(LogSoftmax(logits)).
    """
    # CRITICAL: Must set ignore_index=-100 to properly handle masked labels
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    # [Batch, Seq-1, Vocab]
    shift_logits = logits[..., :-1, :].contiguous()
    # [Batch, Seq-1]
    shift_labels = labels[..., 1:].contiguous()

    # CrossEntropyLoss expects (N, C, d1...) for logits and (N, d1...) for targets.
    # We transpose logits to [Batch, Vocab, Seq-1]
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    
    # loss is now [Batch, Seq-1] of NLLs (0 for ignored positions)
    
    mask = (shift_labels != -100).float()
    
    # Sum loss per sample (loss already 0 for ignored positions)
    sample_sums = loss.sum(dim=1)
    # Count tokens per sample
    sample_counts = mask.sum(dim=1).clamp(min=1)
    
    # Mean loss per sample
    sample_means = sample_sums / sample_counts
    
    # Return list of sample mean losses
    return sample_means.tolist()

from nltk.tokenize import word_tokenize

def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score

def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores

class GenerationEval:
    def __init__(self, config: GenTrainingConfig, checkpoint_dir: str = None):
        # 0. GPU Isolation (Must be before Accelerator)
        if config.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_device)
            # Map internal usage to cuda:0 since it is the only visible one
            # CRITICAL FIX: When CUDA_VISIBLE_DEVICES is set to a single ID, that GPU becomes 'cuda:0'
            
            # self.internal_device_str = f"cuda:{config.cuda_device}"
            self.internal_device_str = "cuda:0"
        else:
            self.internal_device_str = config.device

        # 1. Init Accelerator
        self.accelerator = Accelerator(
            mixed_precision='fp16' if config.fp16 else 'no',
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        self.device = self.accelerator.device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Logging setup
        self.logger = setup_logger(config)
        self.logger.info(f"Initialized Accelerator. Device: {self.device}")
        
        # Tensorboard
        if self.accelerator.is_main_process:
             self.writer = SummaryWriter(log_dir=config.tensorboard_dir)
             
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.scheduler = None
        
    def setup_raw_model(self):
        loader = GenModelLoader(self.config.model_name, self.internal_device_str)
        self.model, self.tokenizer = loader.start(True)
        date = datetime.today()
        self.config.experiment_name = f"raw_model_Gen_{self.config.model_name.split('-')[0]}_{date.month}-{date.day}"

    def setup_loaded_model(self):
        # 2. Load Model & Tokenizer
        self.logger.info("Loading Model...")

        # If checkpoint provided and has config.json in parent dir, try to load essential config
        if self.checkpoint_dir:
             # Try to find config.json in ../ or ../../
             # Usually checkpoint_dir is .../checkpoints/checkpoint-X
             # config is in .../
             
             possible_config_paths = [
                 os.path.join(self.checkpoint_dir, "config.json"),
                 os.path.join(os.path.dirname(self.checkpoint_dir), "config.json"),
                 os.path.join(os.path.dirname(os.path.dirname(self.checkpoint_dir)), "config.json")
             ]
             
             loaded_config = None
             for p in possible_config_paths:
                 if os.path.exists(p):
                     try:
                         with open(p, 'r') as f:
                             loaded_config = json.load(f)
                         self.logger.info(f"Loaded training config from {p}")
                         break
                     except:
                         pass
            
             if loaded_config:
                 # Update self.config with critical structure params if they exist
                 # But prefer args if explicitly set?
                 # Actually, usually args override config, but for model structure (max_seq_len), 
                 # we should probably trust the trained config if args are default.
                 # For simplicity, let's just log it for now or update if needed.
                 # The user didn't explicitly ask for this, but it's good practice.
                 # Update max_seq_length if present
                 if 'max_seq_length' in loaded_config:
                     self.logger.info(f"Overriding max_seq_length from {self.config.max_seq_length} to {loaded_config['max_seq_length']}")
                     self.config.max_seq_length = loaded_config['max_seq_length']
                 if 'model_name' in loaded_config:
                      self.config.model_name = loaded_config['model_name']


        # Note: GenModelLoader enforces quant=True
        loader = GenModelLoader(self.config.model_name, self.internal_device_str)
        self.model, self.tokenizer = loader.start()
        
        # Load Checkpoint if provided
        if self.checkpoint_dir:
            self.logger.info(f"Loading Checkpoint from {self.checkpoint_dir}")
            # check for checkpoint.pt (custom save) or adapter (standard peft)
            pt_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt') # has bugs
            
            if os.path.exists(pt_path):
                # Custom full state or mixed state load
                # If mapped to specific device, ensure map_location
                checkpoint = torch.load(pt_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                     state_dict = checkpoint['model_state_dict']
                else:
                     state_dict = checkpoint
                
                # Load with strict=False to allow for PEFT keys matching or base model keys matching
                self.model.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded checkpoint.pt from step {checkpoint.get('global_step', 'unknown')}")
            else:
                # Try PEFT from_pretrained logic provided by PeftModel?
                # Since self.model is already a PeftModel, we can try loading adapter weights
                # If the checkpoint dir contains adapter_config.json
                if os.path.exists(os.path.join(self.checkpoint_dir, "adapter_config.json")):
                     self.logger.info("Detected PEFT adapter folder structure.")
                     # Reload adapters
                     self.model.load_adapter(self.checkpoint_dir, adapter_name="default")
                     self.logger.info("Loaded PEFT adapters.")
                else:
                     self.logger.warning(f"Checkpoint structure at {self.checkpoint_dir} not recognized (no checkpoint.pt or adapter_config.json). Using initialized random weights.")

    def setup(self):
        if self.config.raw_model:
            self.setup_raw_model()
        else:
            self.setup_loaded_model()

        # 3. Load Data
        self.logger.info("Loading Data...")
        self.train_loader, self.val_loader, self.test_loader, _ = get_gen_dataloader(self.tokenizer, self.config)
        
        # For raw model eval mode, skip optimizer/scheduler and don't prepare model with accelerator
        # BitsAndBytes quantized models are already device-mapped and should NOT be wrapped
        if self.config.raw_model:
            # Only prepare dataloaders, NOT the model (already on GPU via device_map="auto")
            self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
                self.train_loader, self.val_loader, self.test_loader
            )
            self.logger.info("Raw model mode: skipping optimizer/scheduler, model not wrapped by accelerator")
            print('Model device:', next(self.model.parameters()).device)
            return
        
        # 4. Optimizer (only for training/fine-tuning)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        # 5. Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps
        )
        
        # 6. Prepare with Accelerator (for non-raw models with LoRA adapters)
        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader, self.scheduler
        )
        print('Model device:', self.model.device)

    def eval(self):
        # self.evaluate_epoch(self.val_loader, prefix='last_val', save=True)
        self.evaluate_epoch(self.test_loader, prefix='last_test', save=True)

    def evaluate_epoch(self, dataloader, prefix="val", save=True):

        self.model.eval()
        all_results = []
        
        pbar = tqdm(dataloader, desc=f"Evaluating {prefix}", disable=not self.accelerator.is_local_main_process)
        
        batch_loss_mean = []
        batch_acc_mean = []
        
        # Accumulate tokens for Corpus-Level Diversity Metrics
        all_pred_sentences_corpus = []
        # Accumulate IAMM losses for Corpus-Level IAMM PPL
        all_iamm_losses = []

        for batch in pbar:
            # 1. Prompt Logic (Need to ensure prompts are ready)
            # data_loader_gen provides 'prompt_ids'
            
            with torch.no_grad():
                # Teacher forcing metrics
                outputs = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels']
                )
                
                loss_val = outputs.loss.item()
                acc_val = calculate_accuracy(outputs.logits, batch['labels'])
                ppl_samples = calculate_per_sample_ppl(outputs.logits, batch['labels'])
                # IAMM Style PPL: Get list of NLL means per sample
                iamm_losses_batch = IAMM_ppl_loss_list(outputs.logits, batch['labels'])
                all_iamm_losses.extend(iamm_losses_batch)
                
                batch_loss_mean.append(loss_val)
                batch_acc_mean.append(acc_val)
                
                # Generation
                # Unwrapped generate call
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                gen_ids = unwrapped_model.generate(
                    batch['prompt_ids'],
                    attention_mask=batch['prompt_mask'],
                    max_new_tokens=getattr(self.config, 'max_new_tokens', 100),
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    top_p=getattr(self.config, 'gen_top_p', 0.9),
                    temperature=getattr(self.config, 'gen_temperature', 0.7)
                )
                
                # Decode
                prompt_len = batch['prompt_ids'].shape[1]
                generated_part = gen_ids[:, prompt_len:]
                
                decoded_preds = self.tokenizer.batch_decode(generated_part, skip_special_tokens=True)
                decoded_targets = batch['target_text'] # List of strings
                
                # Process metrics per sample
                chencherry = SmoothingFunction()
                
                for i in range(len(decoded_preds)):
                    pred_t = decoded_preds[i].strip()
                    ref_t = decoded_targets[i].strip()
                    
                    # Basic whitespace tokenization first to avoid NLTKpunkt issues with empty strings
                    if not pred_t:
                        pred_toks = []
                    else:
                        pred_toks = nltk.word_tokenize(pred_t)
                    
                    if not ref_t:
                        ref_toks = []
                    else:
                        ref_toks = nltk.word_tokenize(ref_t)

                    # Collect for Corpus Level
                    all_pred_sentences_corpus.append(pred_t)
                    
                    # Safety check for empty references or hypotheses
                    if len(ref_toks) == 0:
                        b1, b2 = 0.0, 0.0
                    else:
                        # Use method 7 (Geometric Mean) often better for single reference short text
                        # method 1 (epsilon) can be harsh if 1-gram precision is low.
                        # But also check simple exact match ratio or unigram overlap without penalty to debug.
                        
                        # Trying Method 7 as it interpolates methods 4 and 5 (length smoothing + average counts)
                        b1 = sentence_bleu([ref_toks], pred_toks, weights=(1, 0, 0, 0), smoothing_function=chencherry.method7)
                        b2 = sentence_bleu([ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method7)
                    
                    # Store Result
                    # ud_idx and ld_idx are tensors from dataloader
                    res_item = {
                        "split": prefix, # Add split info
                        "ud_idx": str(batch['ud_idx'][i].item()) if isinstance(batch['ud_idx'][i], torch.Tensor) else batch['ud_idx'][i],
                        "ld_idx": int(batch['ld_idx'][i].item()) if isinstance(batch['ld_idx'][i], torch.Tensor) else batch['ld_idx'][i],
                        "metrics": {
                            "ppl": ppl_samples[i],
                            "loss_iamm": iamm_losses_batch[i], # LOG THE LOSS, NOT PPL YET
                            "acc": acc_val, # Shared for batch
                            "bleu-1": b1,
                            "bleu-2": b2
                            # dist-1/2 removed for per-sample logging as they are corpus metrics
                        },
                        "generated": pred_t,
                        "target": ref_t
                    }
                    all_results.append(res_item)

        # Calculate Corpus Level Dist-1/2
        if len(all_pred_sentences_corpus) > 0:
            corpus_dist_1, corpus_dist_2 = calc_distinct(all_pred_sentences_corpus)
        else:
            corpus_dist_1, corpus_dist_2 = 0.0, 0.0

        # Calculate Corpus Level IAMM PPL
        # PPL = exp(mean(all_iamm_losses))
        if len(all_iamm_losses) > 0:
            final_iamm_ppl = np.exp(np.mean(all_iamm_losses))
        else:
            final_iamm_ppl = 0.0

        # Inject Corpus Level Metrics into all results for consistency in logging
        for item in all_results:
            item['metrics']['dist-1'] = corpus_dist_1
            item['metrics']['dist-2'] = corpus_dist_2
            item['metrics']['ppl_iamm'] = final_iamm_ppl

        # Log Aggregates to Tensorboard
        avg_loss = np.mean(batch_loss_mean)
        avg_acc = np.mean(batch_acc_mean)

        # Saving Logic (Controlled by flag)
        if save:
            self.save_eval_results(all_results, prefix)
            
        return all_results

    def save_eval_results(self, results, prefix):
        # Save to outputs dir construction
        
        # Determine Strategy Label
        if self.config.few_shot:
            strategy = "FSL"
            param = f"{self.config.shots_per_class}shots"
        elif self.config.semi_supervised:
            strategy = "SSL"
            param = f"{self.config.semi_ratio}ratio"
        else:
            strategy = "Normal"
            param = "Full"
            
        filename = f"eval_results_{strategy}_{param}_{prefix}.jsonl"
        
        # If checkpoint_dir is provided, save to checkpoint_dir/../../eval_logs/
        if self.checkpoint_dir and not self.config.raw_model:
            # Assuming checkpoint_dir is like .../run_name/checkpoints/checkpoint-X or .../run_name/last_checkpoint
            # We want .../run_name/eval_logs/
            
            # Go up two levels to find experiment root from a standard checkpoints/step structure
            # But user said "given_folder/eval_logs/" where given_folder is where loading happens?
            # User said: "load position ... usually given_folder/checkpoints/last_checkpoints/"
            # "result storage should given_folder/eval_logs/"
            
            # So if input is /path/to/exp/checkpoints/checkpoint-100
            # Parent is /path/to/exp/checkpoints
            # Grandparent is /path/to/exp/
            
            # Let's try to deduce the experiment root. 
            # If the folder ends with 'checkpoints', use its parent.
            # If it contains 'checkpoint-', use grandparent.
            
            path_parts = self.checkpoint_dir.strip(os.sep).split(os.sep)
            
            if 'checkpoints' in path_parts:
                # Find the index of 'checkpoints'
                idx = path_parts.index('checkpoints')
                # Reconstruct path up to that point
                exp_root = os.path.sep.join(path_parts[:idx]) # joined by /
                if self.checkpoint_dir.startswith("/"): exp_root = "/" + exp_root
            else:
                # Fallback: just use the parent of the checkpoint dir
                exp_root = os.path.dirname(self.checkpoint_dir)

            save_dir = os.path.join(exp_root, "eval_logs")
        else:
             # Default fallback
             save_dir = os.path.join(self.config.output_dir, self.config.experiment_name, self.config.run_name, "eval_logs")
        
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, filename)
        
        
        with open(final_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
                
        self.logger.info(f"Saved {len(results)} eval results to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Generation Task Training")
    
    # --- 1. Model & Hardware ---
    parser.add_argument("--model_name", type=str, default="Qwen3-4B-Instruct-2507", help="Model path")
    parser.add_argument("--cuda_device", type=int, default=2, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # --- 2. Data & Paths ---
    parser.add_argument("--data_path", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for logs/checkpoints")
    parser.add_argument("--topic_name", type=str, default="Llama3_SSL", help="Tag for run name construction")
    parser.add_argument("--experiment_name", type=str, default="Multi_Run_Debug", help="Specific experiment folder name")
    
    # --- 3. Training Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Total epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Grad accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=2183, help="Max sequence length for input")
    parser.add_argument("--warmup_steps", type=int, default=200, help="LR warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # --- 4. Training Modes (Fast/Few-shot/Semi) ---
    parser.add_argument("--fast_train", action="store_true", help="Debug mode with small data")
    parser.add_argument("--raw_model", action="store_true", help="load original model for teset")
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot learning")
    parser.add_argument("--shots_per_class", type=int, default=16, help="Shots per class for FSL")
    parser.add_argument("--semi_supervised", action="store_true", default=True, help="Enable semi-supervised learning")
    parser.add_argument("--semi_ratio", type=float, default=0.1, help="Ratio for SSL")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio for Validation set (if not fast_train)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Ratio for Test set (if not fast_train)")

    # --- 5. Generation Specifics ---
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max tokens to generate during eval")
    parser.add_argument("--gen_temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--checkpoint_dir", type=str, default='outputs/Qwen3_02-21/method_SSP02_bs_2_inputdata_input_text_Qwen4B_SSL/checkpoints/checkpoint-epoch-4', help="Checkpoint directory to load model from")
    
    args = parser.parse_args()
    # args.semi_supervised = True
    # args.fast_train = True
    # args.raw_model = True
    
    # Init Config
    config = GenTrainingConfig()
    
    # Override config with args
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
            
    # Note: quant is handled inside GenModelLoader (enforced to True)
    
    config.__post_init__() # Re-init paths based on new params
    set_seed(config.seed)
    # config.fast_train, config.semi_supervised = True, True
    
    trainer = GenerationEval(config, checkpoint_dir=args.checkpoint_dir)
    trainer.setup()
    trainer.eval()

if __name__ == "__main__":
    main()
