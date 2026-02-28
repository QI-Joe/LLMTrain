"""
Evaluation Script for SFTTrainer-trained Generation Models

Computes custom metrics:
- Perplexity (PPL)
- IAMM-style PPL
- BLEU-1, BLEU-2
- Distinct-1, Distinct-2
- Token Accuracy

Reuses data loading from ZGeneration.
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
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ZGeneration.data_loader_gen import load_gen_data, GenerationDataset
from src_Gen_SFTTrainer.config_sft import SFTTrainerConfig
from utils_llama3 import set_seed

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def calc_distinct_n(n: int, candidates: list, print_score: bool = False) -> float:
    """Calculate Distinct-N metric"""
    ngram_dict = {}
    total = 0
    candidates = [word_tokenize(c) for c in candidates]
    
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i + n])
            ngram_dict[ngram] = 1
            total += 1
    
    score = len(ngram_dict) / (total + 1e-16)
    if print_score:
        print(f"Distinct-{n}: {score * 100:.2f}%")
    return score


def calc_distinct(candidates: list, print_score: bool = True) -> tuple:
    """Calculate Distinct-1 and Distinct-2"""
    d1 = calc_distinct_n(1, candidates, print_score)
    d2 = calc_distinct_n(2, candidates, print_score)
    return d1, d2


def calculate_accuracy(logits, labels):
    """Calculate token-level accuracy"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    preds = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    correct = (preds == shift_labels) & mask
    if mask.sum() == 0:
        return 0.0
    return (correct.sum().float() / mask.sum().float()).item()


def calculate_per_sample_ppl(logits, labels):
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
    
    return per_sample_ppl.tolist()


def iamm_ppl_loss_list(logits, labels):
    """
    IAMM-style per-sample NLL for corpus PPL calculation.
    Returns list of mean NLL per sample.
    Final PPL = exp(mean(all_sample_losses))
    """
    # CRITICAL: Must set ignore_index=-100 to properly handle masked labels
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    mask = (shift_labels != -100).float()
    
    sample_sums = loss.sum(dim=1)  # loss already has 0 for ignored positions
    sample_counts = mask.sum(dim=1).clamp(min=1)
    sample_means = sample_sums / sample_counts
    
    return sample_means.tolist()


def load_model(model_path: str, adapter_path: str = None, device: str = "cuda"):
    """Load model with optional PEFT adapter"""
    
    # Resolve model path
    dl_path = os.path.expanduser(r'~/Documents/LLModel')
    if '/' not in model_path or len(model_path.split("/")) <= 1:
        model_path = os.path.join(dl_path, model_path)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Left pad for generation
    
    # Quantization - use bfloat16 to match training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model with bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load PEFT adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer


def prepare_eval_data_from_logs(
    tokenizer,
    experiment_dir: str,
    split: str = "test",
) -> "GenerationDataset":
    """
    Reconstruct the evaluation GenerationDataset from the saved eval_logs.

    During training, train_sft.py saves:
      <experiment_dir>/eval_logs/test_block_indices.json  – exact block ids of the test split
      <experiment_dir>/config.json                        – full experiment config

    This function re-creates the EXACT same test GenerationDataset without
    going through the random-split logic again.
    """
    eval_logs_dir = os.path.join(experiment_dir, "eval_logs")
    config_path   = os.path.join(experiment_dir, "config.json")

    # Load config to know data_path, prompt_key, max_seq_length
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in: {experiment_dir}")
    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)

    data_path     = cfg_dict.get("data_path", "./data")
    prompt_key    = cfg_dict.get("prompt_key", "input_text")
    max_seq_len   = int(cfg_dict.get("max_seq_length", 2048))

    # Load block indices for the requested split
    idx_file = os.path.join(eval_logs_dir, f"{split}_block_indices.json")
    if not os.path.exists(idx_file):
        raise FileNotFoundError(
            f"Block-index file not found: {idx_file}\n"
            f"Make sure you ran train_sft.py >= this version which saves test_block_indices.json."
        )
    with open(idx_file, 'r') as f:
        block_indices = json.load(f)

    # Load raw data
    data_dir = os.path.join(data_path, 'gen_task')
    data, _ = load_gen_data(data_dir)

    ds = GenerationDataset.from_block_indices(
        data, tokenizer, max_seq_len, block_indices, prompt_key
    )
    return ds


def prepare_eval_data(tokenizer, config: SFTTrainerConfig, split: str = "test"):
    """Prepare evaluation dataset"""
    data_dir = os.path.join(config.data_path, 'gen_task')
    data, _ = load_gen_data(data_dir)
    
    full_dataset = GenerationDataset(data, tokenizer, config.max_seq_length, prompt_key=config.prompt_key)
    all_block_idx = list(full_dataset.block_map.keys())
    
    # Get test blocks (last portion after train/val)
    if config.few_shot:
        from ZGeneration.data_loader_gen import sample_few_shot_blocks
        train_blocks = sample_few_shot_blocks(data, config.shots_per_class)
    elif config.semi_supervised:
        from ZGeneration.data_loader_gen import sample_semi_supervised
        train_blocks = sample_semi_supervised(np.array(all_block_idx), config.semi_ratio)
    else:
        train_blocks = all_block_idx
    
    remaining = sorted(list(set(all_block_idx) - set(train_blocks)))
    mid = len(remaining) // 2
    
    if split == "val":
        target_blocks = remaining[:mid]
    else:  # test
        target_blocks = remaining[mid:]
    
    eval_ds = GenerationDataset.from_block_indices(
        data, tokenizer, config.max_seq_length, target_blocks, config.prompt_key
    )
    
    return eval_ds


def evaluate(model, tokenizer, dataset: GenerationDataset, config: SFTTrainerConfig, split: str = "test"):
    """Run full evaluation with all metrics"""
    
    device = next(model.parameters()).device
    all_results = []
    all_iamm_losses = []
    all_pred_sentences = []
    
    batch_losses = []
    batch_accs = []
    
    print(f"\nEvaluating {len(dataset)} samples...")
    
    for idx in tqdm(range(len(dataset)), desc=f"Eval {split}"):
        sample = dataset[idx]
        
        # To device
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        labels = sample['labels'].unsqueeze(0).to(device)
        prompt_ids = sample['prompt_ids'].unsqueeze(0).to(device)
        prompt_mask = sample['prompt_mask'].unsqueeze(0).to(device)
        target_text = sample['target_text']
        ud_idx = sample['ud_idx']
        ld_idx = sample['ld_idx']
        
        with torch.no_grad():
            # Forward pass for loss/ppl/acc
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss_val = outputs.loss.item()
            acc_val = calculate_accuracy(outputs.logits, labels)
            ppl_sample = calculate_per_sample_ppl(outputs.logits, labels)[0]
            iamm_loss = iamm_ppl_loss_list(outputs.logits, labels)[0]
            
            batch_losses.append(loss_val)
            batch_accs.append(acc_val)
            all_iamm_losses.append(iamm_loss)
            
            # Generation
            gen_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=config.gen_do_sample,
                top_p=config.gen_top_p,
                temperature=config.gen_temperature,
            )
            
            # Decode
            prompt_len = prompt_ids.shape[1]
            generated_part = gen_ids[:, prompt_len:]
            pred_text = tokenizer.decode(generated_part[0], skip_special_tokens=True).strip()
            
            all_pred_sentences.append(pred_text)
            
            # BLEU
            chencherry = SmoothingFunction()
            pred_toks = word_tokenize(pred_text) if pred_text else []
            ref_toks = word_tokenize(target_text) if target_text else []
            
            if len(ref_toks) == 0:
                b1, b2 = 0.0, 0.0
            else:
                b1 = sentence_bleu([ref_toks], pred_toks, weights=(1, 0, 0, 0), smoothing_function=chencherry.method7)
                b2 = sentence_bleu([ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method7)
            
            result = {
                "split": split,
                "ud_idx": str(ud_idx.item()) if isinstance(ud_idx, torch.Tensor) else ud_idx,
                "ld_idx": int(ld_idx.item()) if isinstance(ld_idx, torch.Tensor) else ld_idx,
                "metrics": {
                    "ppl": ppl_sample,
                    "loss": loss_val,
                    "acc": acc_val,
                    "bleu-1": b1,
                    "bleu-2": b2,
                },
                "generated": pred_text,
                "target": target_text,
            }
            all_results.append(result)
    
    # Corpus-level metrics
    dist_1, dist_2 = calc_distinct(all_pred_sentences, print_score=True)
    iamm_ppl = np.exp(np.mean(all_iamm_losses)) if all_iamm_losses else 0.0
    
    # Add corpus metrics to all results
    for item in all_results:
        item['metrics']['dist-1'] = dist_1
        item['metrics']['dist-2'] = dist_2
        item['metrics']['ppl_iamm'] = iamm_ppl
    
    # Summary
    avg_loss = np.mean(batch_losses)
    avg_acc = np.mean(batch_accs)
    avg_ppl = np.mean([r['metrics']['ppl'] for r in all_results])
    avg_b1 = np.mean([r['metrics']['bleu-1'] for r in all_results])
    avg_b2 = np.mean([r['metrics']['bleu-2'] for r in all_results])
    
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({split})")
    print("=" * 50)
    print(f"Samples:     {len(all_results)}")
    print(f"Avg Loss:    {avg_loss:.4f}")
    print(f"Avg PPL:     {avg_ppl:.4f}")
    print(f"IAMM PPL:    {iamm_ppl:.4f}")
    print(f"Avg Acc:     {avg_acc:.4f}")
    print(f"Avg BLEU-1:  {avg_b1:.4f}")
    print(f"Avg BLEU-2:  {avg_b2:.4f}")
    print(f"Distinct-1:  {dist_1:.4f}")
    print(f"Distinct-2:  {dist_2:.4f}")
    print("=" * 50)
    
    return all_results


def save_results(results: list, output_dir: str, split: str):
    """Save evaluation results to JSONL"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"eval_results_{split}.jsonl")
    
    with open(filepath, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(results)} results to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT-trained Generation Model")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen3-4B-Instruct-2507")
    parser.add_argument("--cuda_device", type=int, default=0)

    # === Data loading: two modes ===
    # Mode A (recommended): provide experiment_dir — adapter path, model name, seq length,
    #   and exact test split are all auto-resolved from the saved config.json and
    #   eval_logs/test_block_indices.json inside that directory.
    parser.add_argument(
        "--experiment_dir", type=str, default=None,
        help="Path to a training output dir containing config.json and "
             "eval_logs/test_block_indices.json. Activates Mode A automatically."
    )
    # Mode B (legacy): re-run split logic from scratch (may drift if seed/ratio changed)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--prompt_key", type=str, default="input_text",
                        help="[Mode B] Prompt field key in the dataset (e.g. input_text, ws_prompt).")
    parser.add_argument("--max_seq_length", type=int, default=2183)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument("--semi_supervised", action="store_true", default=True)
    parser.add_argument("--semi_ratio", type=float, default=0.1)

    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write eval_results_*.jsonl. "
                             "Defaults to <experiment_dir>/eval_logs (Mode A) or "
                             "./outputs/eval_results (Mode B).")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Use sampling for generation (default: True). "
                             "Pass --no_do_sample for greedy decoding.")
    parser.add_argument("--no_do_sample", dest="do_sample", action="store_false")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # GPU (module-level env var already set by early parser; this covers legacy path)
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Determine data-loading mode
    # Mode A: experiment_dir provided  →  auto-resolve everything
    # Mode B: experiment_dir absent    →  use explicit CLI flags
    # ------------------------------------------------------------------
    use_eval_logs = args.experiment_dir is not None

    # Resolve output dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            os.path.join(args.experiment_dir, "eval_logs")
            if use_eval_logs else "./outputs/eval_results"
        )

    # Build a minimal config (only generation hyper-params needed at eval time)
    config = SFTTrainerConfig()
    config.max_new_tokens  = args.max_new_tokens
    config.gen_temperature = args.temperature
    config.gen_top_p       = args.top_p
    config.gen_do_sample   = args.do_sample

    adapter_path = None

    if use_eval_logs:
        # ---- Mode A: reconstruct exact split from saved block indices ----
        cfg_path = os.path.join(args.experiment_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                saved_cfg = json.load(f)
            # Use CLI --model_name if explicitly provided, else fall back to saved config
            if args.model_name == parser.get_default("model_name"):
                config.model_name = saved_cfg.get("model_name", args.model_name)
            else:
                config.model_name = args.model_name
            config.max_seq_length = int(saved_cfg.get("max_seq_length", 2048))
        else:
            print(f"[WARNING] config.json not found in {args.experiment_dir}, using CLI model_name.")
            config.model_name = args.model_name

        candidate = os.path.join(args.experiment_dir, "checkpoints", "final_model")
        if os.path.isdir(candidate):
            adapter_path = candidate
        else:
            print(f"[WARNING] Adapter not found at {candidate} — evaluating base model without fine-tuning!")
    else:
        # ---- Mode B: legacy re-split (use with caution — split may differ from training) ----
        config.model_name      = args.model_name
        config.data_path       = args.data_path
        config.prompt_key      = args.prompt_key
        config.max_seq_length  = args.max_seq_length
        config.few_shot        = args.few_shot
        config.shots_per_class = args.shots_per_class
        config.semi_supervised = args.semi_supervised
        config.semi_ratio      = args.semi_ratio
        config.__post_init__()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(config.model_name, adapter_path)

    # Prepare data
    print("Preparing evaluation data...")
    if use_eval_logs:
        eval_dataset = prepare_eval_data_from_logs(tokenizer, args.experiment_dir, args.split)
    else:
        eval_dataset = prepare_eval_data(tokenizer, config, args.split)
    print(f"Loaded {len(eval_dataset)} samples for {args.split}")

    # Evaluate
    results = evaluate(model, tokenizer, eval_dataset, config, args.split)

    # Save
    save_results(results, output_dir, args.split)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
