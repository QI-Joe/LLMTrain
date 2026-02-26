"""
src-Gen-SFTTrainer: SFTTrainer-based Generation Fine-tuning

This module provides an alternative training approach using TRL's SFTTrainer
instead of the custom training loop in ZGeneration.

Key Features:
- Uses SFTTrainer from TRL library
- QLoRA (4-bit quantization + LoRA) support
- Reuses data loading from ZGeneration (FSL/SSL support)
- Custom evaluation with BLEU, Distinct, IAMM PPL metrics

Files:
- config_sft.py: Configuration dataclass
- train_sft.py: Main training script with SFTTrainer
- eval_sft.py: Evaluation script with custom metrics
- run_sft_train.sh: Shell script for training
- run_sft_eval.sh: Shell script for evaluation

Usage:
    # Training
    python -m src_Gen_SFTTrainer.train_sft --model_name Qwen3-4B-Instruct-2507 --semi_supervised
    
    # Evaluation
    python -m src_Gen_SFTTrainer.eval_sft --adapter_path ./outputs/.../final_model

Dependencies:
    pip install trl peft bitsandbytes accelerate datasets transformers nltk
"""

from src_Gen_SFTTrainer.config_sft import SFTTrainerConfig, get_sft_config

__all__ = ['SFTTrainerConfig', 'get_sft_config']
