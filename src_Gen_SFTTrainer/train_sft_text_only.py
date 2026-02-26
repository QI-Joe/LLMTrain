"""
SFTTrainer-based Training Script for Generation Task
Uses TRL's SFTTrainer with QLoRA for efficient fine-tuning.

Reuses data loading from ZGeneration for FSL/SSL support.
"""
import os
import sys, json
import argparse

# ============================================================================
# CRITICAL: Parse CUDA device BEFORE importing torch or any module that imports torch!
# This ensures GPU isolation works properly with device_map="auto"
# ============================================================================
def _parse_cuda_device_early():
    """Early argument parsing for CUDA device isolation"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda_device", type=int, default=None)
    args, _ = parser.parse_known_args()
    return args.cuda_device

_cuda_device = _parse_cuda_device_early()
if _cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda_device)
    print(f"[GPU Isolation] CUDA_VISIBLE_DEVICES={_cuda_device} (visible as cuda:0)")
# ============================================================================

# NOW it's safe to import torch and torch-dependent modules
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

# Add parent to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Reuse ZGeneration data loading — text-only variant
from ZGeneration.new_data_loader.data_loader_gen_text_only import (
    load_gen_data, gen_loader_warp, GenerationDataset
)
from src_Gen_SFTTrainer.config_sft import SFTTrainerConfig
from utils_llama3 import set_seed


def create_hf_text_dataset(gen_dataset: GenerationDataset) -> Dataset:
    """
    Convert GenerationDataset to a HuggingFace Dataset containing plain-text strings.

    For the text-only pipeline the dataset returns a single 'text' field which is
    the full formatted conversation (system + history + final assistant turn) produced
    by apply_chat_template.  SFTTrainer tokenises the text itself and computes causal
    LM loss over the whole sequence.  No pre-computed labels or input_ids are needed.

    Non-string metadata fields (emotion, ud_idx, ld_idx) are kept out of the HF
    Dataset to avoid collation errors; they are retained in the raw GenerationDataset
    for later evaluation use.
    """
    texts = []
    for i in range(len(gen_dataset)):
        sample = gen_dataset[i]   # returns {'text': str, 'prompt_text': str, ...}
        texts.append({"text": sample["text"]})
    return Dataset.from_list(texts)


def load_model_and_tokenizer(config: SFTTrainerConfig):
    """
    Load model with quantization and prepare for LoRA.
    """
    # Resolve model path
    model_path = config.model_name
    dl_path = os.path.expanduser(r'~/Documents/LLModel')
    if '/' not in config.model_name or len(config.model_name.split("/")) <= 1:
        model_path = os.path.join(dl_path, config.model_name)
        print(f"Loading local model: {model_path}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'  # SFTTrainer typically uses right padding
    
    # Quantization Config
    bnb_config = None
    if config.use_4bit:
        # Use bfloat16 for compute dtype to match bf16 training
        compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )
    
    # Load Model - use same dtype as training precision
    model_dtype = torch.bfloat16 if config.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    
    # Prepare for LoRA if using PEFT
    if config.use_peft:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


class PPLLoggerCallback(TrainerCallback):
    """
    Zero-overhead PPL logging via TrainerCallback.

    How it works:
    - Trainer already computes eval_loss = mean masked cross-entropy (tokens where
      label==-100 are excluded by nn.CrossEntropyLoss(ignore_index=-100) inside
      the model's forward). No logit accumulation needed.
    - PPL = exp(eval_loss) is derived in one float operation per eval epoch.
    - prediction_loss_only stays True (default), so logits are NEVER kept in memory.

    Why the previous approaches failed:
    - compute_metrics:          forces prediction_loss_only=False → Trainer accumulates
                                ALL logits from ALL batches into one numpy array before
                                calling the function → OOM (N × L × V × dtype bytes).
    - eval_accumulation_steps=1: flushes logits to CPU every step, but the internal
                                np.concatenate loop is O(n²) in time as the buffer grows,
                                causing eval time to increase with each evaluation.
    """
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            ppl = round(float(torch.exp(torch.tensor(eval_loss)).item()), 4)
            metrics["eval_ppl"] = ppl
            # Also push to trainer logs so TensorBoard / console picks it up
            if state.log_history:
                state.log_history[-1]["eval_ppl"] = ppl
            print(f"  eval_ppl = {ppl:.4f}  (exp({eval_loss:.4f}))")


def get_peft_config(config: SFTTrainerConfig) -> LoraConfig:
    """Create LoRA configuration"""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )


def prepare_datasets(tokenizer, config: SFTTrainerConfig):
    """
    Load and prepare datasets using the text-only data loader.
    Returns HuggingFace Dataset objects (text strings only) for SFTTrainer,
    plus the raw GenerationDataset objects and the detected slide_windows value.
    """
    data_dir = os.path.join(config.data_path, 'gen_task')
    raw_data, max_len, slide_windows = load_gen_data(data_dir)
    config.max_seq_length = max_len

    _, _, _, raw_ds = gen_loader_warp(raw_data, tokenizer, config)
    train_ds, val_ds, test_ds = raw_ds

    # Convert to plain-text HF Datasets — SFTTrainer handles tokenisation
    train_hf = create_hf_text_dataset(train_ds)
    val_hf   = create_hf_text_dataset(val_ds)
    test_hf  = create_hf_text_dataset(test_ds)

    return train_hf, val_hf, test_hf, raw_ds, slide_windows


def main():
    parser = argparse.ArgumentParser(description="SFTTrainer for Generation Task")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen3-4B-Instruct-2507")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Data
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument('--fast_train', action="store_true")
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.05)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # Training Modes
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument("--semi_supervised", action="store_true", default=True)
    parser.add_argument("--semi_ratio", type=float, default=0.1)
    
    # LoRA / PEFT
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--no_peft", action="store_true", help="Disable LoRA")

    # Slide window (used in output folder naming only; must match the preloader)
    parser.add_argument("--slide_windows", type=int, default=None,
                        help="Slide-window size used by the preloader (auto-detected from pkl if not set)")

    # Naming
    parser.add_argument("--topic_name", type=str, default="SFT", help="Tag for run name")
    
    args = parser.parse_args()
    
    # Note: GPU isolation already done at module load time (before torch import)
    # This ensures proper device mapping
    
    # Build Config
    config = SFTTrainerConfig()
    
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    # CRITICAL: Map batch_size to per_device_*_batch_size BEFORE __post_init__
    # because __post_init__ syncs batch_size FROM per_device_train_batch_size
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
        config.per_device_eval_batch_size = args.batch_size

    # Disable LoRA when --no_peft is passed
    if args.no_peft:
        config.use_peft = False

    # Map num_epochs to num_train_epochs
    if args.num_epochs is not None:
        config.num_train_epochs = args.num_epochs
    
    # Reinit paths (this also syncs batch_size back from per_device_train_batch_size)
    # topic_name is baked in before __post_init__ so the folder name reflects TextOnly+SW.
    # If --slide_windows was explicitly supplied, embed it now; otherwise it will be
    # patched after the data is loaded and the pkl filename is inspected.
    if args.slide_windows is not None:
        config.topic_name = f"TextOnly_SW{args.slide_windows}"
    else:
        config.topic_name = "TextOnly"  # will be updated after load_gen_data
    config.__post_init__()
    set_seed(config.seed)
    
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Run: {config.run_name}")
    print(f"Output: {config.full_output_dir}")
    print(f"Batch Size: {config.per_device_train_batch_size} (train), {config.per_device_eval_batch_size} (eval)")
    print(f"Grad Accum: {config.gradient_accumulation_steps}")
    print(f"Epochs: {config.num_train_epochs}")
    print("=" * 60)
    
    # Load Model
    print("\n[1/4] Loading Model...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare Datasets
    print("\n[2/4] Preparing Datasets...")
    train_dataset, val_dataset, test_dataset, raw_ds, slide_windows = prepare_datasets(tokenizer, config)

    # If slide_windows was auto-detected (not supplied via CLI), patch the run/output names now
    if args.slide_windows is None and slide_windows > 0:
        config.topic_name = f"TextOnly_SW{slide_windows}"
        config.__post_init__()   # rebuilds run_name and full_output_dir with the updated topic
        print(f"[Naming] Auto-detected SW={slide_windows} → output: {config.full_output_dir}")

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} samples")
    
    # LoRA Config
    peft_config = get_peft_config(config) if config.use_peft else None

    # Text-only pipeline: SFTTrainer tokenises the 'text' field itself.
    # dataset_text_field MUST be "text" so SFTTrainer applies the chat template +
    # causal LM loss over the full sequence.  Setting it to None would disable
    # SFTTrainer's own tokenisation and expect pre-computed input_ids/labels tensors.
    config.dataset_text_field = "text"

    # Create SFTConfig (includes max_seq_length, packing, dataset_text_field)
    print("\n[3/4] Setting up Trainer...")
    sft_config = config.to_sft_config()

    # PPL is logged via PPLLoggerCallback (zero overhead):
    #   eval_loss  = masked cross-entropy (Trainer computes this by default,
    #                prediction_loss_only=True, no logit accumulation)
    #   eval_ppl   = exp(eval_loss)  — one float op per epoch in the callback
    # Both appear in console logs and TensorBoard automatically.

    # SFTTrainer - uses SFTConfig which has all training args + SFT-specific params
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[PPLLoggerCallback()],
    )
    
    # Train
    print("\n[4/4] Starting Training...")
    trainer.train()
    
    # Save final model to checkpoints folder (matching ZGeneration structure)
    print("\nSaving final model...")
    final_ckpt_path = os.path.join(config.checkpoint_dir, "final_model")
    os.makedirs(final_ckpt_path, exist_ok=True)
    trainer.save_model(final_ckpt_path)
    tokenizer.save_pretrained(final_ckpt_path)
    
    # Save test dataset reference for later evaluation (in eval_logs folder)
    eval_logs_dir = os.path.join(config.full_output_dir, "eval_logs")
    os.makedirs(eval_logs_dir, exist_ok=True)
    test_dataset.save_to_disk(os.path.join(eval_logs_dir, "test_dataset"))

    # Save test block indices (from GenerationDataset) so eval_sft.py can
    # reconstruct the EXACT same split without re-running the data pipeline.
    _, _, test_ds_raw = raw_ds  # raw GenerationDataset objects
    test_block_indices = sorted(set(
        test_ds_raw.data['ud_idx'][i] for i in test_ds_raw.indices
    ))
    import json as _json
    with open(os.path.join(eval_logs_dir, "test_block_indices.json"), 'w') as _f:
        _json.dump(test_block_indices, _f)
    print(f"  Saved {len(test_block_indices)} test block indices -> {eval_logs_dir}/test_block_indices.json")
    
    # Save config for reproducibility
    config_path = os.path.join(config.full_output_dir, "config.json")
    config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, list, type(None))) else v 
                   for k, v in config.__dict__.items()}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\nTraining Complete!")
    print(f"  Output:      {config.full_output_dir}")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"  Final Model: {final_ckpt_path}")


if __name__ == "__main__":
    main()
