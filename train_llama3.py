"""
训练脚本骨架 - Llama3情感分类微调
Training Script Skeleton for Llama3 Emotion Classification

这是一个骨架模板，主要展示训练流程结构
详细的实现细节请参考 markdown/Llama3微调指南.md
"""
import os, sys
import torch
import argparse
from tqdm import tqdm
from config_llama3 import get_config, TrainingConfig
from data_loader_llama3 import get_dataloader
from utils_llama3 import ResultRecorder, setup_logger, set_seed
from transformers import AutoTokenizer, AutoModel
from model_loader import LlamaClassification, LlamaModelDownload
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from BackPG_Test import get_snapshot_saver


class Llama3EmotionClassifier:
    """Llama3情感分类器训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger(config)
        self.recorder = ResultRecorder(config)
        # self.param_update_check = get_snapshot_saver(config.learning_rate)
        
        # 初始化组件
        self.model_name = self.config.model_name
        self.tokenizer: AutoTokenizer = None
        self.model: LlamaClassification = None
        self.optimizer: AdamW = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = torch.amp.GradScaler(enabled=True)
        
    def setup(self):
        """设置所有组件"""
        self.logger.info("=" * 50)
        self.logger.info("Setting up Llama3 Emotion Classifier")
        self.logger.info("=" * 50)
        
        # 1. 加载分词器和模型
        self.load_model()
        
        # 2. 加载数据
        self.load_data()
        
        # 3. 设置优化器和调度器
        self.setup_optimizer()
        
        self.logger.info("Setup completed!")
        
    def load_model(self):
        """
        加载Llama3模型和分词器
        详细说明见 markdown/Llama3微调指南.md - 模型加载部分
        """
        self.logger.info(f"Loading model: {self.config.model_name} on device {self.config.device}")
        # 修复：正确调用LlamaModelDownload
        model_downloader = LlamaModelDownload(self.config.model_name, self.config.device, self.config.quant)
        self.LLModel, self.tokenizer = model_downloader.start()
        self.model = LlamaClassification(
            self.config.hidden_size, 
            model=self.LLModel, 
            tokenizer=self.tokenizer,
        )
        self.tokenizer.padding_side = 'right'
        
        # 打印模型设备信息
        model_device = next(self.model.parameters()).device
        self.logger.info(f"Model loaded successfully on device: {model_device}")
        self.logger.info(f"Target device: {self.config.device}")
        
        
    def load_data(self):
        """
        加载EmpatheticDialogues数据集
        详细说明见 markdown/Llama3微调指南.md - 数据加载部分
        """
        self.logger.info("Loading data...")
        
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(
            config=self.config,
            tokenizer=self.tokenizer
        )
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        
    def setup_optimizer(self):
        """
        设置优化器和学习率调度器
        详细说明见 markdown/Llama3微调指南.md - 优化器配置部分
        """
        self.logger.info("Setting up optimizer...")
        
        # TODO: 配置优化器
        no_decay = ['bias', 'LayerNorm.weight', 'RMSNorm.weight']
        optimizer_grouped = [
            {
                'params': [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        self.optimizer = AdamW(optimizer_grouped, lr=self.config.learning_rate, betas=(0.9, 0.95))
    
        # TODO: 配置学习率调度器
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps)
        
        self.logger.info("Optimizer setup completed")
        
        # 打印可训练参数信息
        trained_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trained_params / all_params
        self.logger.info(
            f"Trainable params: {trained_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {trainable_percent:.2f}%"
        )
    
    def is_last_epoch_step(self, given_epoch, given_step, len_data: int):
        if given_epoch>= (self.config.num_epochs-1) and given_step >= (len_data-1):
            return True
        return False
    
    def _move_batch_to_device(self, batch):
        batch["input_ids"] = batch["input_ids"].to(self.config.device)
        batch["attention_mask"]= batch["attention_mask"].to(self.config.device)
        batch["labels"]= batch["labels"].to(self.config.device)
        return batch
        
    def train_epoch(self, epoch):
        """
        训练一个epoch
        详细说明见 markdown/Llama3微调指南.md - 训练循环部分
        
        Args:
            epoch: 当前epoch编号
        """
        self.model.train()
        total_loss, global_step = 0, 0
        accumulation_steps = self.config.gradient_accumulation_steps
        len_dataloader = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # global_step = len(self.train_loader) * epoch + step + 1
            
            # 将batch移动到GPU
            batch = self._move_batch_to_device(batch)
            
            with torch.amp.autocast(device_type='cuda'):
                logits, loss = self.model.forward(batch)
            
            # 归一化loss（重要！）
            loss: torch.Tensor = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            
            # 只在累积步数达到时更新
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # 计算实际loss（恢复到原始尺度）
                actual_loss = loss.item() * accumulation_steps
                total_loss += actual_loss
                
                # 计算global_step（只在更新时递增）
                global_step = epoch * (len(self.train_loader) // accumulation_steps) + (step + 1) // accumulation_steps
                
                # 更新进度条
                pbar.set_postfix({"loss": f"{actual_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                
                # 定期记录
                if global_step % self.config.log_interval == 0:
                    self.recorder.log_scalar("train/loss", actual_loss, global_step)
                    self.recorder.log_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)
                    # current_tensor_dict = {name: params.clone().detach().cpu() for name, params in self.model.named_parameters()}
                    # self.param_update_check.save_snapshot(epoch=epoch, step=step, tensor_dict=current_tensor_dict)
                
                # 定期评估
                if global_step % self.config.eval_interval == 0:
                    self.evaluate(global_step)
                    self.model.train()
                
                # 定期保存
                if global_step % self.config.save_interval == 0 or self.is_last_epoch_step(epoch, step, len_dataloader):
                    self.save_checkpoint(global_step)
            # if global_step > 100: sys.exit(0)
        # 计算平均loss（基于实际更新次数）
        num_updates = (len(self.train_loader) + accumulation_steps - 1) // accumulation_steps
        avg_loss = total_loss / num_updates
        return avg_loss
    
    def evaluate(self, global_step):
        """
        在验证集上评估模型
        详细说明见 markdown/Llama3微调指南.md - 评估部分
        
        Args:
            global_step: 当前全局步数
        """
        self.logger.info(f"Evaluating at step {global_step}...")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # 将batch移动到GPU
                batch = self._move_batch_to_device(batch)
                
                # TODO: 前向传播
                logits, loss = self.model(batch)
                
                # TODO: 获取预测结果
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                total_loss += loss.item()
        
        # TODO: 计算评估指标
        metrics = self.compute_metrics(all_preds, all_labels)
        avg_loss = total_loss / len(self.val_loader)
        
        # 记录评估结果
        self.recorder.log_metrics(metrics, global_step, prefix="val")
        self.recorder.log_scalar("val/loss", avg_loss, global_step)
        
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        self.logger.info(f"Metrics: {metrics}")
        
    def compute_metrics(self, preds, labels):
        """
        计算评估指标
        详细说明见 markdown/Llama3微调指南.md - 评估指标部分
        
        Args:
            preds: 预测标签
            labels: 真实标签
            
        Returns:
            metrics: 包含各种指标的字典
        """
        
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0)
        }
        
        return metrics
    
    def save_checkpoint(self, global_step):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 修复：保存完整的检查点
        checkpoint = {
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, 'checkpoint.pt'))
        
        # 只保存tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        """
        完整训练流程
        详细说明见 markdown/Llama3微调指南.md - 训练流程部分
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Training")
        self.logger.info("=" * 50)
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练一个epoch
            avg_loss = self.train_epoch(epoch)
            
            self.logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            # 每个epoch结束后评估 - 修复：使用正确的步数
            self.evaluate((epoch + 1) * len(self.train_loader))
        
        # 训练结束后在测试集上评估
        self.test()
        
        # # 关闭参数监控系统
        # self.param_update_check.end_process()
        
        self.logger.info("Training completed!")
        
    def test(self):
        """
        在测试集上进行最终评估
        详细说明见 markdown/Llama3微调指南.md - 测试部分
        """
        self.logger.info("=" * 50)
        self.logger.info("Testing on test set")
        self.logger.info("=" * 50)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # 将batch移动到GPU
                batch = self._move_batch_to_device(batch)
                
                # TODO: 前向传播
                logits, loss = self.model(batch)  # 修复：model返回(logits, loss)元组
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # 计算最终指标
        metrics = self.compute_metrics(all_preds, all_labels)
        self.logger.info(f"Test Metrics: {metrics}")
        self.recorder.save_final_results(metrics)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Llama3 Emotion Classification Fine-tuning")
    
    # Model Parameters
    parser.add_argument("--model_name", type=str, default="Llama-3.3-8B-Instruct", help="Model name or path", choices=['Llama-3.3-8B-Instruct','Qwen3-4B-Instruct-2507', 'Qwen3-1.7B']) 
    parser.add_argument("--topic_name", type=str, default="LlamaTest")
    # Data Parameters
    parser.add_argument("--hidden_size", type=int, default=4096, choices=[4096, 2560, 2048])
    parser.add_argument("--data_path", type=str, default="./data/emotion_cls", help="Data path")
    parser.add_argument("--prompt_key", type=str, default="ws_prompt", 
                       choices=["ws_prompt", "wo_prompt"],
                       help="ws_prompt=with situation, wo_prompt=without situation")
    parser.add_argument("--dialogue_window", type=int, default=3, help="Dialogue window size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Training Parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Few-shot Learning
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot learning")
    parser.add_argument("--shots_per_class", type=int, default=16, 
                       help="Number of blocks per emotion class in few-shot mode")
    
    # Semi-supervised Learning
    parser.add_argument("--semi_supervised", action="store_true", help="Enable semi-supervised learning")
    parser.add_argument("--semi_ratio", type=float, default=0.1, 
                       help="Ratio of labeled data in semi-supervised mode")
    
    # Fast Training Mode
    parser.add_argument("--fast_train", action="store_true", 
                       help="Use only 20% of val/test data for faster training")
    
    # Logging & Checkpointing
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    # Device & Precision
    parser.add_argument("--cuda_device", type=int, default=1, help="CUDA device ID (0, 1, 2...)")
    parser.add_argument("--quant", action="store_false", help="Enable mixed precision training")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 加载默认配置
    config = get_config()
    
    # 用命令行参数覆盖配置
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    # 重新计算依赖的参数
    config.__post_init__()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建训练器并开始训练
    trainer = Llama3EmotionClassifier(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
