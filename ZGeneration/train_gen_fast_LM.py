import sys
import os
import argparse
import torch
import numpy as np
import json
import nltk
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu

# Add parent directory to path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ZGeneration.config_gen import GenTrainingConfig
from ZGeneration.quick_dataloader import get_dataloader
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
    # Without this, loss for -100 positions is undefined (index out of vocab range)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # [Batch, Seq-1]
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    # With ignore_index=-100, loss for ignored positions is already 0
    # Mask is still needed to count valid tokens
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

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, batch):
        # batch is list of dicts from Dataset
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        prompt_ids = [item['prompt_ids'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        
        # Pad input_ids (right)
        input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        
        # Attention Mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids_padded != self.pad_token_id).long()
        
        # Pad labels (right) with -100
        labels_padded = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Pad prompt_ids (left) for generation
        # pad_sequence pads to the right. To pad left, reverse, pad, reverse.
        prompt_ids_reversed = [t.flip(0) for t in prompt_ids]
        prompt_ids_padded_reversed = rnn_utils.pad_sequence(prompt_ids_reversed, batch_first=True, padding_value=self.pad_token_id)
        prompt_ids_padded = prompt_ids_padded_reversed.flip(1)
        prompt_mask = (prompt_ids_padded != self.pad_token_id).long()
        
        # Handle optional keys safely
        ud_idxs = [item.get('ud_idx', -1) for item in batch]
        ld_idxs = [item.get('ld_idx', -1) for item in batch]
        
        # Convert to tensor if list is numbers
        if ud_idxs and isinstance(ud_idxs[0], int):
            ud_idxs = torch.tensor(ud_idxs)
        if ld_idxs and isinstance(ld_idxs[0], int):
            ld_idxs = torch.tensor(ld_idxs)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
            "prompt_ids": prompt_ids_padded,
            "prompt_mask": prompt_mask,
            "target_text": target_texts,
            "ud_idx": ud_idxs,
            "ld_idx": ld_idxs
        }

class GenerationTrainer:
    def __init__(self, config: GenTrainingConfig):
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

    def setup(self):
        # 2. Load Model & Tokenizer
        self.logger.info("Loading Model...")
        # Note: GenModelLoader enforces quant=True
        loader = GenModelLoader(self.config.model_name, self.internal_device_str, self.accelerator)
        self.model, self.tokenizer = loader.start()
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 3. Load Data
        self.logger.info("Loading Data...")
        self.train_dataset, self.test_dataset = get_dataloader(self.tokenizer)
        
        collator = DataCollator(self.tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collator, num_workers=self.config.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collator, num_workers=self.config.num_workers)
        
        # 4. Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        # 5. Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps
        )
        
        # 6. Prepare with Accelerator
        self.model, self.optimizer, self.train_loader, self.test_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.test_loader, self.scheduler
        )
        
    def train(self):
        self.logger.info("Starting Training...")
        global_step = 0
        total_batches = len(self.train_loader)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(enumerate(self.train_loader), total=total_batches, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
            
            for step, batch in pbar:
                with self.accelerator.accumulate(self.model):
                    # Forward
                    outputs = self.model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    if self.accelerator.sync_gradients:
                         global_step += 1
                         if self.accelerator.is_main_process:
                             self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                             
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                
                # Intermediate Validation condition (80% of batches)
                # if (step + 1) == val_trigger_step:
                #      self.logger.info(f"Running Intermediate Validation at step {step+1}/{total_batches}...")
                #      self.evaluate_epoch(epoch, self.val_loader, f"val_inter_ep{epoch}", save=False)
                #      self.model.train()

            # --- End of Epoch Evaluation ---
            self.logger.info(f"Epoch {epoch} completed. Running End-of-Epoch Evaluation...")
            
            # Use standard validation logic unless it is the final epoch
            is_final_epoch = (epoch == self.config.num_epochs - 1)
            
            if not is_final_epoch:
                pass
                # self.evaluate_epoch(epoch, self.val_loader, "val", save=False)
            else:
                 # During final epoch, run both val and test and save together
                 self.logger.info("Final Epoch detected. Running Combined Validation & Test Evaluation...")
                #  val_results = self.evaluate_epoch(epoch, self.val_loader, "val", save=True)
                 test_results = self.evaluate_epoch(epoch, self.test_loader, "test", save=True)
                 
                 # Combine results
                 # Mark split in results to distinguish?
                #  for res in val_results: res['split'] = 'val'
                #  for res in test_results: res['split'] = 'test'
                 
                #  combined_results = val_results + test_results
                #  self.save_eval_results(combined_results, epoch, "final_combined")
                 
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_path = os.path.join(self.config.checkpoint_dir, f"checkpoint-epoch-{epoch}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(ckpt_path)
            self.tokenizer.save_pretrained(ckpt_path)
    
    def save_eval_results(self, results, epoch, prefix):
        # Save to outputs dir
        # Filename: eval_results_epoch_{epoch}_rank_{rank}.jsonl
        filename = f"eval_results_epoch_{prefix}_{epoch}_rank_{self.accelerator.process_index}.jsonl"
        out_path = os.path.join(self.config.tensorboard_dir, filename) # Reuse param_update_check dir or just TB dir?
        # Config output_dir usually better
        save_dir = os.path.join(self.config.output_dir, self.config.experiment_name, self.config.run_name, "eval_logs")
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, filename)
        
        
        with open(final_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
                
        self.logger.info(f"Saved {len(results)} eval results to {final_path}")

    def evaluate_epoch(self, epoch, dataloader, prefix="val", save=True):
        self.model.eval()
        all_results = []
        
        pbar = tqdm(dataloader, desc=f"Evaluating {prefix}", disable=not self.accelerator.is_local_main_process)
        
        batch_loss_mean = []
        batch_acc_mean = []
        
        # Accumulate tokens for Corpus-Level Diversity Metrics
        all_pred_tokens_corpus = []

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
                    all_pred_tokens_corpus.extend(pred_toks)
                    
                    # Safety check for empty references or hypotheses
                    if len(ref_toks) == 0:
                        b1, b2 = 0.0, 0.0
                    else:
                        # method 1 (epsilon) can be harsh if 1-gram precision is low.
                        # But also check simple exact match ratio or unigram overlap without penalty to debug.
                        b1 = sentence_bleu([ref_toks], pred_toks, weights=(1, 0, 0, 0), )
                        b2 = sentence_bleu([ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0),)
                    
                    # Store Result
                    # ud_idx and ld_idx are tensors from dataloader
                    res_item = {
                        "epoch": epoch,
                        "split": prefix, # Add split info
                        "metrics": {
                            "ppl": ppl_samples[i],
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
        if len(all_pred_tokens_corpus) > 0:
            corpus_dist_1 = len(set(all_pred_tokens_corpus)) / len(all_pred_tokens_corpus)
            bigrams = list(zip(all_pred_tokens_corpus, all_pred_tokens_corpus[1:]))
            corpus_dist_2 = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0
        else:
            corpus_dist_1, corpus_dist_2 = 0.0, 0.0

        # Inject Corpus Level Metrics into all results for consistency in logging
        for item in all_results:
            item['metrics']['dist-1'] = corpus_dist_1
            item['metrics']['dist-2'] = corpus_dist_2

        # Log Aggregates to Tensorboard
        avg_loss = np.mean(batch_loss_mean)
        avg_acc = np.mean(batch_acc_mean)
        
        if self.accelerator.is_main_process:
             self.writer.add_scalar(f'{prefix}/Loss', avg_loss, epoch)
             self.writer.add_scalar(f'{prefix}/Acc', avg_acc, epoch)
             self.writer.add_scalar(f'{prefix}/Dist-1', corpus_dist_1, epoch) # Log Corpus dist
             self.writer.add_scalar(f'{prefix}/Dist-2', corpus_dist_2, epoch) # Log Corpus dist
             # Calculate avg metrics from main process view (approximation)
             if len(all_results) > 0:
                 avg_b1 = np.mean([r['metrics']['bleu-1'] for r in all_results])
                 self.writer.add_scalar(f'{prefix}/BLEU-1', avg_b1, epoch)

        # Saving Logic (Controlled by flag)
        if save:
            self.save_eval_results(all_results, epoch, prefix)
            
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Generation Task Training")
    
    # --- 1. Model & Hardware ---
    parser.add_argument("--model_name", type=str, default="Llama-3.3-8B-Instruct", help="Model path")
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
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for input")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # --- 4. Training Modes (Fast/Few-shot/Semi) ---
    parser.add_argument("--fast_train", action="store_true", help="Debug mode with small data")
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot learning")
    parser.add_argument("--shots_per_class", type=int, default=16, help="Shots per class for FSL")
    parser.add_argument("--semi_supervised", action="store_true", default=True, help="Enable semi-supervised learning")
    parser.add_argument("--semi_ratio", type=float, default=0.1, help="Ratio for SSL")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio for Validation set (if not fast_train)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Ratio for Test set (if not fast_train)")

    # --- 5. Generation Specifics ---
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max tokens to generate during eval")
    parser.add_argument("--gen_temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Init Config
    config = GenTrainingConfig()
    
    # Override config with args
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
            
    # Note: quant is handled inside GenModelLoader (enforced to True)
    
    config.__post_init__() # Re-init paths based on new params
    set_seed(config.seed)
    
    trainer = GenerationTrainer(config)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    main()
