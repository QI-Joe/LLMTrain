"""
Evaluation Script for Text-Only SFTTrainer-trained Models

Key Features:
1. Explicit CUDA device control (no auto mapping)
2. Manual tokenization with label masking for PPL computation
3. Separate generation pass for BLEU/DIST metrics
4. Loads model from experiment directory, saves results to eval_logs/

Metrics computed:
- PPL / IAMM_PPL (Logits-based, masked for assistant-only)
- BLEU-1, BLEU-2 (Text-based)
- Distinct-1, Distinct-2 (Text-based)
- Token Accuracy
"""
import os
import sys
import json
import argparse

# ============================================================================
# CRITICAL: Parse CUDA device BEFORE importing torch
# ============================================================================
def _parse_cuda_device_early():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda_device", type=int, default=None)
    args, _ = parser.parse_known_args()
    return args.cuda_device

_cuda_device = _parse_cuda_device_early()
if _cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda_device)
    print(f"[GPU Isolation] CUDA_VISIBLE_DEVICES={_cuda_device} (visible as cuda:0)")
# ============================================================================

import torch
import numpy as np
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ZGeneration.new_data_loader.data_loader_gen_text_only import (
    load_gen_data, GenerationDataset
)
from utils_llama3 import set_seed

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


class TextOnlyEvaluator:
    """
    Evaluator for text-only SFT-trained models.
    
    Handles both Logits-based (PPL) and Generation-based (BLEU/DIST) metrics
    with explicit GPU control.
    """
    
    def __init__(
        self,
        model_path: str,
        adapter_path: str = None,
        device_id: int = 0,
        use_4bit: bool = True,
        use_bf16: bool = True,
    ):
        """
        Initialize evaluator with explicit device control.
        
        Args:
            model_path: Path to base model
            adapter_path: Path to PEFT adapter (optional)
            device_id: CUDA device ID (explicit, no auto mapping)
            use_4bit: Whether to use 4-bit quantization
            use_bf16: Whether to use bfloat16
        """
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"[Evaluator] Using device: {self.device}")
        
        # Resolve model path
        dl_path = os.path.expanduser(r'~/Documents/LLModel')
        if '/' not in model_path or len(model_path.split("/")) <= 1:
            model_path = os.path.join(dl_path, model_path)
        
        print(f"[Evaluator] Loading model from: {model_path}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        bnb_config = None
        if use_4bit:
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model - explicit device mapping, NOT auto
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        # For quantized models, we need device_map but can restrict to single GPU
        if use_4bit:
            # Create explicit device map for single GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map={"": self.device},  # Force all layers to specific device
                trust_remote_code=True,
                torch_dtype=model_dtype,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=model_dtype,
            ).to(self.device)
        
        # Load PEFT adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            print(f"[Evaluator] Loading PEFT adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
        print(f"[Evaluator] Model loaded successfully")
    
    def tokenize_with_labels(self, full_text: str, prompt_text: str):
        """
        Tokenize full conversation and create labels with prompt masking.
        
        The key insight: we need to compute loss ONLY on the assistant response,
        not on the user prompt. This is done by:
        1. Tokenize the full text (prompt + response)
        2. Tokenize the prompt only
        3. Create labels = input_ids.clone()
        4. Mask labels[:prompt_length] = -100
        
        Returns:
            input_ids, attention_mask, labels (all tensors)
        """
        # Tokenize full sequence
        full_enc = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        input_ids = full_enc['input_ids']
        attention_mask = full_enc['attention_mask']
        
        # Tokenize prompt only to find where response starts
        prompt_enc = self.tokenizer(
            prompt_text,
            return_tensors='pt',
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        prompt_len = prompt_enc['input_ids'].shape[1]
        
        # Create labels with masking
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # Mask prompt portion
        
        return input_ids, attention_mask, labels, prompt_len
    
    def calculate_per_sample_ppl(self, logits, labels):
        """Calculate per-sample perplexity"""
        # CRITICAL: Must set ignore_index=-100 to properly handle masked labels
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        mask = (shift_labels != -100).float()
        
        counts = mask.sum(dim=1)
        sums = loss.sum(dim=1)  # loss already has 0 for ignored positions
        per_sample_loss = sums / counts.clamp(min=1)
        per_sample_ppl = torch.exp(per_sample_loss)
        
        return np.average(per_sample_ppl).item(), np.average(per_sample_loss).item()
    
    def calculate_accuracy(self, logits, labels):
        """Calculate token-level accuracy"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        preds = torch.argmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        correct = (preds == shift_labels) & mask
        if mask.sum() == 0:
            return 0.0
        return (correct.sum().float() / mask.sum().float()).item()
    
    @staticmethod
    def calc_distinct_n(n: int, candidates: list) -> float:
        """Calculate Distinct-N metric"""
        ngram_dict = {}
        total = 0
        candidates = [word_tokenize(c) for c in candidates]
        
        for sentence in candidates:
            for i in range(len(sentence) - n + 1):
                ngram = tuple(sentence[i:i + n])
                ngram_dict[ngram] = 1
                total += 1
        
        return len(ngram_dict) / (total + 1e-16)
    
    def evaluate(
        self,
        dataset: GenerationDataset,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        split: str = "test",
    ):
        """
        Run full evaluation with PPL and Generation metrics.
        
        The evaluation has two passes:
        1. Forward pass: Compute PPL/Loss/Accuracy (logits-based)
        2. Generation pass: Generate text for BLEU/DIST computation
        """
        all_results = []
        all_iamm_losses = []
        all_pred_sentences = []
        all_ref_sentences = []
        
        print(f"\n[Evaluator] Evaluating {len(dataset)} samples on {split} split...")
        
        for idx in tqdm(range(len(dataset)), desc=f"Eval {split}"):
            sample = dataset[idx]
            
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            labels = sample['labels']
            ud_idx, ld_idx = sample['ud_idx'], sample['ld_idx']
            raw_text = sample['input_data']
            emotion = sample['emotion']
            
            # -----------------------------------------------------------------
            # Pass 1: Forward pass for PPL/Loss/Accuracy
            # -----------------------------------------------------------------
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss_val = outputs.loss.item()
                acc_val = self.calculate_accuracy(outputs.logits, labels)
                ppl_vals, loss_list = self.calculate_per_sample_ppl(outputs.logits, labels)
                ppl_sample = ppl_vals
                iamm_loss = loss_list
                all_iamm_losses.append(iamm_loss)
            
            # -----------------------------------------------------------------
            # Pass 2: Generation for BLEU/DIST
            # -----------------------------------------------------------------
            self.tokenizer.padding_side = 'left'  # Critical for generation!
            
            b1_list, b2_list = list(), list()
            for idx, item in enumerate(raw_text):
                if 'system' in item.keys() or 'assistant' in item.keys(): continue
                current_ctx = raw_text[:idx]
                input_text = self.tokenizer.apply_chat_template(
                    current_ctx,
                    tokenize=False,
                    add_generation_prompt = True
                )
                prompt_enc = self.tokenizer(
                    input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=2048,
                )
                prompt_ids = prompt_enc['input_ids'].to(self.device)
                prompt_mask = prompt_enc['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                    )
            
                # Decode generated text
                gen_len = prompt_ids.shape[1]
                generated_part = gen_ids[:, gen_len:]
                pred_text = self.tokenizer.decode(generated_part[0], skip_special_tokens=True).strip()
                all_pred_sentences.append(pred_text)
                
                # Extract reference text (target) from full_text
                # Decode the labels (non -100 tokens) to get the target
                target_text = raw_text[idx+1]['content']
                all_ref_sentences.append(target_text)
                
                # -----------------------------------------------------------------
                # Compute BLEU scores
                # -----------------------------------------------------------------
                chencherry = SmoothingFunction()
                pred_toks = word_tokenize(pred_text) if pred_text else []
                ref_toks = word_tokenize(target_text) if target_text else []
                
                if len(ref_toks) == 0:
                    b1, b2 = 0.0, 0.0
                else:
                    b1 = sentence_bleu(
                        [ref_toks], pred_toks, weights=(1, 0, 0, 0),
                        smoothing_function=chencherry.method7
                    )
                    b2 = sentence_bleu(
                        [ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=chencherry.method7
                    )
                b1_list.append(b1)
                b2_list.append(b2)
            
            result = {
                "split": split,
                "ud_idx": str(ud_idx) if not isinstance(ud_idx, str) else ud_idx,
                "ld_idx": int(ld_idx) if isinstance(ld_idx, np.integer) else ld_idx,
                "emotion": emotion,
                "metrics": {
                    "ppl": ppl_sample,
                    "loss": loss_val,
                    "acc": acc_val,
                    "bleu-1": sum(b1)/len(b1),
                    "bleu-2": sum(b2)/len(b2),
                },
                "generated": pred_text,
                "target": target_text,
            }
            all_results.append(result)
        
        # -----------------------------------------------------------------
        # Corpus-level metrics
        # -----------------------------------------------------------------
        dist_1 = self.calc_distinct_n(1, all_pred_sentences)
        dist_2 = self.calc_distinct_n(2, all_pred_sentences)
        iamm_ppl = np.exp(np.mean(all_iamm_losses)) if all_iamm_losses else 0.0
        
        # Add corpus metrics to all results
        for item in all_results:
            item['metrics']['dist-1'] = dist_1
            item['metrics']['dist-2'] = dist_2
            item['metrics']['ppl_iamm'] = iamm_ppl
        
        # Summary
        avg_loss = np.mean([r['metrics']['loss'] for r in all_results])
        avg_acc = np.mean([r['metrics']['acc'] for r in all_results])
        avg_ppl = np.mean([r['metrics']['ppl'] for r in all_results])
        avg_b1 = np.mean([r['metrics']['bleu-1'] for r in all_results])
        avg_b2 = np.mean([r['metrics']['bleu-2'] for r in all_results])
        
        print("\n" + "=" * 60)
        print(f"Evaluation Results ({split})")
        print("=" * 60)
        print(f"Samples:     {len(all_results)}")
        print(f"Avg Loss:    {avg_loss:.4f}")
        print(f"Avg PPL:     {avg_ppl:.4f}")
        print(f"IAMM PPL:    {iamm_ppl:.4f}")
        print(f"Avg Acc:     {avg_acc:.4f}")
        print(f"Avg BLEU-1:  {avg_b1:.4f}")
        print(f"Avg BLEU-2:  {avg_b2:.4f}")
        print(f"Distinct-1:  {dist_1:.4f}")
        print(f"Distinct-2:  {dist_2:.4f}")
        print("=" * 60)
        
        return all_results


def load_test_dataset_from_experiment(experiment_dir: str, tokenizer, prompt_key: str = "input_text"):
    """
    Load test dataset from experiment directory using saved block indices.
    
    The training script saves:
    - eval_logs/test_block_indices.json: List of block IDs for test split
    - config.json: Full experiment config
    """
    eval_logs_dir = os.path.join(experiment_dir, "eval_logs")
    config_path = os.path.join(experiment_dir, "config.json")
    
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in: {experiment_dir}")
    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)
    
    data_path = cfg_dict.get("data_path", "./data")
    max_seq_len = int(cfg_dict.get("max_seq_length", 2048))
    prompt_key = cfg_dict.get("prompt_key", prompt_key)
    
    # Load block indices
    idx_file = os.path.join(eval_logs_dir, "test_block_indices.json")
    if not os.path.exists(idx_file):
        raise FileNotFoundError(
            f"Block index file not found: {idx_file}\n"
            f"Make sure the training script saved test_block_indices.json"
        )
    with open(idx_file, 'r') as f:
        block_indices = json.load(f)
    
    # Load raw data
    data_dir = os.path.join(data_path, 'gen_task')
    raw_data, _, _ = load_gen_data(data_dir)
    
    # Create dataset from saved block indices
    dataset = GenerationDataset.from_block_indices(
        raw_data, tokenizer, max_seq_len, block_indices, prompt_key
    )
    
    print(f"[Data] Loaded {len(dataset)} test samples from {len(block_indices)} blocks")
    return dataset


def save_results(results: list, output_dir: str, split: str):
    """Save evaluation results to JSONL"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"eval_results_{split}.jsonl")
    
    with open(filepath, 'w') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Also save summary
    summary_path = os.path.join(output_dir, f"eval_summary_{split}.json")
    summary = {
        "split": split,
        "num_samples": len(results),
        "avg_ppl": float(np.mean([r['metrics']['ppl'] for r in results])),
        "iamm_ppl": float(results[0]['metrics']['ppl_iamm']) if results else 0.0,
        "avg_loss": float(np.mean([r['metrics']['loss'] for r in results])),
        "avg_acc": float(np.mean([r['metrics']['acc'] for r in results])),
        "avg_bleu1": float(np.mean([r['metrics']['bleu-1'] for r in results])),
        "avg_bleu2": float(np.mean([r['metrics']['bleu-2'] for r in results])),
        "distinct1": float(results[0]['metrics']['dist-1']) if results else 0.0,
        "distinct2": float(results[0]['metrics']['dist-2']) if results else 0.0,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[Output] Saved {len(results)} results to {filepath}")
    print(f"[Output] Saved summary to {summary_path}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-Only SFT Model")
    
    # Model / Paths
    parser.add_argument(
        "--experiment_dir", type=str, required=True,
        help="Path to training output dir (contains config.json and checkpoints/)"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Override model name from config. If not set, uses saved config."
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0,
        help="CUDA device ID to use (explicit, no auto mapping)"
    )
    
    # Data
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--prompt_key", type=str, default="input_text")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_do_sample", dest="do_sample", action="store_false",
                        help="Use greedy decoding instead of sampling")
    
    # Quantization
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no_bf16", action="store_true", help="Use fp16 instead of bf16")
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Validate experiment directory
    if not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")
    
    # Load config from experiment
    config_path = os.path.join(args.experiment_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved_cfg = json.load(f)
        model_name = args.model_name or saved_cfg.get("model_name", "Qwen3-4B-Instruct-2507")
    else:
        print(f"[WARNING] config.json not found, using CLI model_name")
        model_name = args.model_name or "Qwen3-4B-Instruct-2507"
    
    # Find adapter path
    adapter_path = os.path.join(args.experiment_dir, "checkpoints", "final_model")
    if not os.path.isdir(adapter_path):
        print(f"[WARNING] Adapter not found at {adapter_path}")
        print(f"[WARNING] Evaluating base model without fine-tuning!")
        adapter_path = None
    
    # Output directory
    output_dir = os.path.join(args.experiment_dir, "eval_logs")
    
    # Initialize evaluator (explicit device control)
    print("\n" + "=" * 60)
    print("Text-Only SFT Model Evaluation")
    print("=" * 60)
    print(f"Experiment: {args.experiment_dir}")
    print(f"Model: {model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Device: cuda:{args.cuda_device}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    evaluator = TextOnlyEvaluator(
        model_path=model_name,
        adapter_path=adapter_path,
        device_id=args.cuda_device,
        use_4bit=not args.no_4bit,
        use_bf16=not args.no_bf16,
    )
    
    # Load test dataset
    print("\n[Data] Loading test dataset...")
    test_dataset = load_test_dataset_from_experiment(
        args.experiment_dir,
        evaluator.tokenizer,
        args.prompt_key,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        dataset=test_dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        split=args.split,
    )
    
    # Save results
    save_results(results, output_dir, args.split)
    
    print("\n[Complete] Evaluation finished!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
