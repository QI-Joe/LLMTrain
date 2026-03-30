"""
配置文件 - Llama3情感分类微调
Configuration for Llama3 Emotion Classification Fine-tuning
"""
import torch
import argparse
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class TrainingConfig:
    """训练配置 Training Configuration"""
    
    # ========== Model Parameters ==========
    model_name: str = "Llama-3.3-8B-Instruct"  # 或者本地路径
    num_emotions: int = 32  # EmpatheticDialogues有32个情感标签
    hidden_size: int = 4096
    
    # ========== Data Parameters ==========
    data_path: str = "./data/emotion_cls"  # 预处理后的数据路径
    prompt_key: str = "ws_prompt"  # 'ws_prompt' (with situation) 或 'wo_prompt'
    dialogue_window: int = 3  # IAMM论文中的对话窗口大小
    max_seq_length: int = 512
    batch_size: int = 4
    num_workers: int = 4
    topic_name: str = "LlamaTest"
    
    # ========== Training Parameters ==========
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # ========== Few-shot Learning Parameters ==========
    few_shot: bool = False  # 是否启用few-shot训练
    shots_per_class: int = 16  # Few-shot: 每个类别的block数
    
    # ========== Semi-supervised Learning Parameters ==========
    semi_supervised: bool = True  # 是否启用semi-supervised训练
    semi_ratio: float = 0.2  # Semi-supervised: 使用10%的数据作为labeled
    
    # ========== Fast Training Mode ==========
    fast_train: bool = True  # 快速训练模式（使用20%的val/test数据）
    quant: bool = False
    accerlator: bool = False
    
    # ========== Logging & Checkpointing ==========
    log_interval: int = 100  # 每100步记录一次loss
    eval_interval: int = 1000  # 每1000步评估一次
    save_interval: int = 1000  # 每1000步保存一次checkpoint
    
    output_dir: str = "./outputs"
    experiment_name: str = f"Llama3_{datetime.today().strftime('%m-%d')}"  
    
    # TensorboardX 记录的参数 (可自定义3个参数)
    param1: str = "method" 
    param2: str = "bs"  # batch_size
    param3: str = "inputdata"  # dialogue_window
    
    # ========== Device ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_device: int = 1  # 指定使用的GPU设备ID（0, 1, 2...）
    fp16: bool = True  # 混合精度训练
    seed: int = 42
    task: str = 'Classification'
    
    # ========== Evaluation Metrics ==========
    metrics: list = None  # ["accuracy", "f1", "precision", "recall"]
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "f1", "precision", "recall"]
        
        # 设置具体的CUDA设备
        if torch.cuda.is_available():
            self.device = f"cuda:{self.cuda_device}"
        
        # 构建实验文件夹名称
        model_simplifed_name = self.model_name.split('-')
        the_method = 'full_train'
        if self.few_shot: the_method="FSL"
        elif self.semi_supervised: the_method = "SSP"
        
        if self.accerlator: self.cuda_device='cpu'
        
        self.experiment_name = f"{model_simplifed_name[0]+self.task}_{datetime.today().strftime('%m-%d')}"
        self.run_name = f"{self.param1}_{the_method}_{self.param2}_{self.batch_size}_{self.param3}_{self.prompt_key}_{self.topic_name}"
        self.tensorboard_dir = f"{self.output_dir}/{self.experiment_name}/{self.run_name}"
        self.checkpoint_dir = f"{self.output_dir}/{self.experiment_name}/{self.run_name}/checkpoints"


def get_config():
    """获取默认配置实例"""
    return TrainingConfig()


if __name__ == "__main__":
    config = get_config()
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    for key, value in config.__dict__.items():
        print(f"{key:30s}: {value}")
    print("=" * 50)
