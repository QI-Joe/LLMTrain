"""
SFTTrainer Configuration for Generation Task
Leverages TRL's SFTTrainer with LoRA/QLoRA
"""
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from trl import SFTConfig

# Import parent config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ZGeneration.config_gen import GenTrainingConfig


@dataclass
class SFTTrainerConfig(GenTrainingConfig):
    """
    SFTTrainer-specific configuration.
    Inherits from GenTrainingConfig for data loading compatibility.
    """
    
    # ========== SFTTrainer Specific ==========
    # These map to TrainingArguments
    output_dir: str = "./outputs"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    # LR Scheduler
    lr_scheduler_type: str = "linear"  # linear, cosine, constant, etc.
    
    # Logging
    logging_steps: int = 10
    logging_dir: str = None  # TensorBoard dir, auto-set in __post_init__
    
    # Evaluation
    eval_strategy: str = "epoch"  # "steps" or "epoch"
    eval_steps: int = 500  # Only used if eval_strategy="steps"
    
    # Saving
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Training Precision
    # NOTE: Use bf16=True for modern GPUs (Ampere+). fp16 with GradScaler doesn't support BFloat16 tensors.
    fp16: bool = False
    bf16: bool = True
    
    # Gradient Settings
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    
    # ========== LoRA/PEFT Settings ==========
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # ========== Quantization ==========
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # Use bfloat16 to match bf16=True
    use_double_quant: bool = True
    
    # ========== SFT Specific ==========
    max_seq_length: int = 2183
    packing: bool = False  # Dataset packing for efficiency
    dataset_text_field: str = None  # Field name for text in dataset
    
    # ========== Generation Evaluation ==========
    eval_generation: bool = True
    max_new_tokens: int = 150
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_do_sample: bool = True
    
    # ========== Custom Fields ==========
    experiment_name: str = None
    run_name: str = None
    topic_name: str = "SFT"  # Tag for run name (like ZGeneration)
    
    def __post_init__(self):
        # Call parent post_init for path setup
        super().__post_init__()
        
        # ================================================================
        # Match ZGeneration folder naming convention:
        # experiment_name: {model}Gen_{date}  (e.g., Qwen3Gen_02-24)
        # run_name: method_{method}_bs_{batch}_inputdata_{prompt_key}_{topic}
        # ================================================================
        
        model_name_short = self.model_name.split('-')[0] if '-' in self.model_name else self.model_name
        date_str = datetime.today().strftime('%m-%d')
        
        # Method label (matching ZGeneration: FSL, SSP, full_train)
        if self.few_shot:
            method = "FSL"
            method_detail = f"FSL{self.shots_per_class}"
        elif self.semi_supervised:
            method = "SSP"
            method_detail = f"SSL{int(self.semi_ratio*100)}"
        else:
            method = "full_train"
            method_detail = "Full"
        
        # Match ZGeneration naming exactly
        self.experiment_name = f"{model_name_short}Gen_{date_str}"
        self.run_name = f"method_{method}_bs_{self.per_device_train_batch_size}_inputdata_{self.prompt_key}_{self.topic_name}"
        
        # Set up directories (matching ZGeneration structure)
        self.full_output_dir = os.path.join(self.output_dir, self.experiment_name, self.run_name)
        self.tensorboard_dir = self.full_output_dir  # TensorBoard logs in run folder
        self.logging_dir = os.path.join(self.full_output_dir, "logs")
        self.checkpoint_dir = os.path.join(self.full_output_dir, "checkpoints")
        
        # Sync batch_size for data loader compatibility
        self.batch_size = self.per_device_train_batch_size

    def to_sft_config(self) -> SFTConfig:
        """
        Create TRL SFTConfig from this config.
        SFTConfig extends TrainingArguments with SFT-specific params like max_seq_length.
        """
        # Ensure directories exist
        os.makedirs(self.full_output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        return SFTConfig(
            output_dir=self.full_output_dir,
            # eval_accumulation_steps = 1,
            
            # ========== Batch & Accumulation ==========
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            
            # ========== Learning Rate ==========
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio if self.warmup_steps == 0 else 0,
            warmup_steps=self.warmup_steps,
            
            # ========== Epochs ==========
            num_train_epochs=self.num_train_epochs,
            
            # ========== Logging ==========
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            logging_strategy="steps",
            report_to=["tensorboard"],
            
            # ========== Evaluation ==========
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps if self.eval_strategy == "steps" else None,
            
            # ========== Saving ==========
            save_strategy=self.save_strategy,
            save_steps=self.save_steps if self.save_strategy == "steps" else None,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            
            # ========== Precision & Optimization ==========
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            gradient_checkpointing=self.gradient_checkpointing,
            
            # ========== SFT Specific (max_seq_length supported here!) ==========
            max_length=self.max_seq_length,
            packing=self.packing,
            dataset_text_field=self.dataset_text_field,
            
            # ========== Misc ==========
            seed=self.seed,
            remove_unused_columns=False,  # Important for custom datasets
            dataloader_num_workers=self.num_workers,
            run_name=self.run_name,
        )


def get_sft_config():
    """Get default SFT config"""
    return SFTTrainerConfig()


"""
def create_training_args(config: SFTTrainerConfig) -> TrainingArguments:
    
    os.makedirs(config.full_output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    return TrainingArguments(
        output_dir=config.full_output_dir,
        
        # Batch & Accumulation
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning Rate
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio if config.warmup_steps == 0 else 0,
        warmup_steps=config.warmup_steps,
        
        # Epochs
        num_train_epochs=config.num_train_epochs,
        
        # Logging
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        report_to=["tensorboard"],
        
        # Evaluation
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        
        # Saving
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy == "steps" else None,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # Precision & Optimization
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Misc
        seed=config.seed,
        remove_unused_columns=False,  # Important for custom datasets
        dataloader_num_workers=config.num_workers,
        
        # Run name for tracking
        run_name=config.run_name,
    )

"""