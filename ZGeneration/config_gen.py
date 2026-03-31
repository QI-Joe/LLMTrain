from dataclasses import dataclass
import sys
import os

# 引用上层目录模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_llama3 import TrainingConfig

@dataclass
class GenTrainingConfig(TrainingConfig):
    """
    生成任务专用配置，继承自通用配置
    Generation-specific configuration inheriting from general TrainingConfig
    """
    # 覆盖默认值 (Override defaults)
    prompt_key: str = "input_text"  # 生成任务特定的Prompt键名
    data_path: str = r'../data/'
    data_dir: str = r'./data/ED'
    device: str = '0'
    
    # 新增生成特定参数 (New generation-specific parameters)
    max_new_tokens: int = 150       # 生成的最大长度
    gen_temperature: float = 0.7    # 采样温度
    gen_top_p: float = 0.9          # Nucleus sampling
    
    # 评估指标开关 (Evaluation metrics switches)
    eval_bleu: bool = True
    eval_rouge: bool = True
    
    # Dataset Split Ratios (For FSL/SSL/Fast options)
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    raw_model: bool = False
    
    task: str = 'Gen'
