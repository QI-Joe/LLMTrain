"""
工具函数 - 结果记录、日志、随机种子等
Utility Functions for Result Recording, Logging, Random Seed, etc.

包含:
1. ResultRecorder - 使用TensorboardX记录训练结果
2. Logger设置
3. 随机种子设置
4. 其他辅助函数

详细说明见 markdown/Llama3微调指南.md
"""
import os
import json
import random
import logging
import numpy as np
import torch
from datetime import datetime
from tensorboardX import SummaryWriter


class ResultRecorder:
    """
    结果记录器
    
    功能:
    1. 使用TensorboardX记录训练过程
    2. 记录loss (每log_interval步)
    3. 记录评估指标 (每eval_interval步)
    4. 保存最终结果到JSON文件
    5. 组织文件结构: Llama3_Day/{param1}_{param2}_{param3}/
    """
    
    def __init__(self, config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 创建输出目录
        os.makedirs(config.tensorboard_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 初始化TensorboardX writer
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        # 记录配置信息
        self.save_config()
        
        # 用于记录最佳结果
        self.best_metrics = {}
        
        print(f"Results will be saved to: {config.tensorboard_dir}")
        
    def save_config(self):
        """保存配置到JSON文件"""
        config_file = os.path.join(self.config.tensorboard_dir, "config.json")
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('_')}
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {config_file}")
    
    def log_scalar(self, tag, value, step):
        """
        记录标量值
        
        Args:
            tag: 标签名称，如 "train/loss", "val/accuracy"
            value: 标量值
            step: 当前步数
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step):
        """
        记录多个标量值
        
        Args:
            tag: 标签名称
            value_dict: 字典，如 {"train": 0.5, "val": 0.6}
            step: 当前步数
        """
        self.writer.add_scalars(tag, value_dict, step)
    
    def log_metrics(self, metrics, step, prefix="val"):
        """
        记录评估指标
        
        Args:
            metrics: 指标字典，如 {"accuracy": 0.85, "f1": 0.80, ...}
            step: 当前步数
            prefix: 前缀，如 "val" 或 "test"
        """
        for metric_name, value in metrics.items():
            self.log_scalar(f"{prefix}/{metric_name}", value, step)
        
        # 更新最佳结果
        for metric_name, value in metrics.items():
            key = f"{prefix}_{metric_name}"
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def log_histogram(self, tag, values, step):
        """
        记录直方图
        
        Args:
            tag: 标签名称
            values: 数值数组
            step: 当前步数
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag, text, step):
        """
        记录文本
        
        Args:
            tag: 标签名称
            text: 文本内容
            step: 当前步数
        """
        self.writer.add_text(tag, text, step)
    
    def log_embedding(self, mat, metadata=None, tag="embedding", step=0):
        """
        记录embedding可视化
        
        Args:
            mat: embedding矩阵 [N, D]
            metadata: 每个点的标签列表
            tag: 标签名称
            step: 当前步数
        """
        self.writer.add_embedding(mat, metadata=metadata, tag=tag, global_step=step)
    
    def save_final_results(self, test_metrics):
        """
        保存最终测试结果
        
        Args:
            test_metrics: 测试集指标字典
        """
        results = {
            "test_metrics": test_metrics,
            "best_val_metrics": self.best_metrics,
            "config": {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('_')},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(self.config.tensorboard_dir, "final_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Final results saved to: {results_file}")
        
        # 同时保存一份简洁的结果摘要
        summary_file = os.path.join(self.config.tensorboard_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("Final Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Test Metrics:\n")
            for k, v in test_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\nBest Validation Metrics:\n")
            for k, v in self.best_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
        
        print(f"Summary saved to: {summary_file}")
    
    def close(self):
        """关闭writer"""
        self.writer.close()


def setup_logger(config):
    """
    设置日志记录器
    
    Args:
        config: 配置对象
        
    Returns:
        logger: 日志记录器
    """
    # 创建logger
    logger = logging.getLogger("Llama3EmotionClassifier")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    log_file = os.path.join(config.tensorboard_dir, "training.log")
    os.makedirs(config.tensorboard_dir, exist_ok=True)  # 修复：创建目录而不是文件路径
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


def set_seed(seed):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数数
        trainable_params: 可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds):
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        formatted_time: 格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_predictions(predictions, labels, output_file, emotion_names=None):
    """
    保存预测结果用于后续分析
    
    Args:
        predictions: 预测标签
        labels: 真实标签
        output_file: 输出文件路径
        emotion_names: 情感名称列表
    """
    results = []
    for pred, label in zip(predictions, labels):
        result = {
            "prediction": int(pred),
            "label": int(label),
            "correct": int(pred) == int(label)
        }
        if emotion_names:
            result["prediction_name"] = emotion_names[int(pred)]
            result["label_name"] = emotion_names[int(label)]
        results.append(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Predictions saved to: {output_file}")


def compute_class_weights(dataset, num_classes):
    """
    计算类别权重用于处理类别不平衡
    
    Args:
        dataset: 数据集
        num_classes: 类别数量
        
    Returns:
        weights: 类别权重张量
    """
    from collections import Counter
    
    # 统计每个类别的样本数
    label_counts = Counter()
    for item in dataset:
        label_counts[item['label']] += 1
    
    # 计算权重 (逆频率)
    total_samples = len(dataset)
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # 避免除零
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def get_lr(optimizer):
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        lr: 当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """
    早停机制
    
    当验证指标在patience个epoch内没有改善时停止训练
    """
    
    def __init__(self, patience=5, mode='max', delta=0.0):
        """
        Args:
            patience: 容忍的epoch数
            mode: 'max' 或 'min'，指标是越大越好还是越小越好
            delta: 最小改善幅度
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Args:
            score: 当前指标值
            
        Returns:
            improved: 是否改善
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


if __name__ == "__main__":
    """
    测试工具函数
    """
    from config_llama3 import get_config
    
    print("=" * 50)
    print("Testing Utility Functions")
    print("=" * 50)
    
    config = get_config()
    
    # 测试日志
    logger = setup_logger(config)
    logger.info("Testing logger")
    
    # 测试ResultRecorder
    recorder = ResultRecorder(config)
    
    # 模拟记录一些数据
    for step in range(0, 1000, 100):
        recorder.log_scalar("train/loss", 1.0 / (step + 1), step)
        
        if step % 200 == 0:
            metrics = {
                "accuracy": 0.5 + step / 2000,
                "f1": 0.4 + step / 2000
            }
            recorder.log_metrics(metrics, step, prefix="val")
    
    # 保存最终结果
    test_metrics = {
        "accuracy": 0.85,
        "f1": 0.83,
        "precision": 0.84,
        "recall": 0.82
    }
    recorder.save_final_results(test_metrics)
    
    recorder.close()
    
    print("\nUtility functions test completed!")
    print(f"Check results in: {config.tensorboard_dir}")
    print("\nTo view tensorboard:")
    print(f"  tensorboard --logdir={config.output_dir}")
