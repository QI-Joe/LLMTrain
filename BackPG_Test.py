import queue, threading
import os, json
import numpy as np

from datetime import datetime
import torch
from torch import Tensor, nn
from typing import Tuple, Dict, Optional, Any, List

class AsyncSnapshotCapture:
    def __init__(self, save_dir: str = r"./outputs/params_update_check", learning_rate: float = 2e-5):
        os.makedirs(save_dir, exist_ok=True)
        self.write_dir = save_dir
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Initialize tracking variables
        self.prev_item: Optional[Tuple[int, int, Dict[str, torch.Tensor]]] = None
        self.learning_rate = learning_rate
        
        # Set threshold based on learning rate
        # Threshold = learning_rate * 10 (经验值：参数变化应该在学习率量级)
        # 使用相对阈值：如果参数变化的平均绝对值 > threshold，则认为参数已更新
        self.threshold = learning_rate * 10.0
        
        # Alternative: 使用更保守的阈值（更小的值意味着更敏感的检测）
        self.threshold = learning_rate * 2
        
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        print(f"Capture system background started... (threshold={self.threshold:.2e})")
        
    def save_snapshot(self, epoch: int, step: int, tensor_dict: Dict[str, Tensor]):
        """保存参数快照到队列（异步处理）"""
        # 确保所有tensor都在CPU上
        cpu_dict = {}
        for key, value in tensor_dict.items():
            if value.is_cuda:
                cpu_dict[key] = value.detach().cpu()
            else:
                cpu_dict[key] = value.detach() if value.requires_grad else value
        self.queue.put((epoch, step, cpu_dict))
    
    def updated_param_detection(self, item: Tuple[int, int, Dict[str, torch.Tensor]]):
        """
        检测参数是否被更新的核心算法
        
        比较策略：
        1. 计算参数的平均绝对变化量 (Mean Absolute Difference)
        2. 如果变化量 > threshold，认为参数已更新
        3. threshold基于learning_rate设定，确保检测敏感度适中
        
        Args:
            item: (epoch, step, tensor_dict)
        
        Returns:
            modified_param_list: 已更新的参数名列表
            steady_param_list: 未更新（冻结）的参数名列表
        """
        epoch, step, tensor_dict = item
        
        # Handle first call - no previous item to compare
        if self.prev_item is None:
            # Return empty changed list and all params as steady (with their sizes)
            steady_list = [(key, tensor.numel()) for key, tensor in tensor_dict.items()]
            return [], steady_list
        
        prev_epoch, prev_step, prev_tensor_dict = self.prev_item
        modified_param_list, steady_param_list = list(), list()
        key_list: List = list(tensor_dict.keys())
        
        for key in key_list:
            # 确保参数存在于两个字典中
            if key not in prev_tensor_dict:
                modified_param_list.append((key, tensor_dict[key].numel()))
                continue
            
            # 计算差异
            current_tensor = tensor_dict[key]
            prev_tensor = prev_tensor_dict[key]
            
            # 确保tensor在CPU上并转为numpy
            if current_tensor.is_cuda:
                current_tensor = current_tensor.cpu()
            if prev_tensor.is_cuda:
                prev_tensor = prev_tensor.cpu()
            
            difference = (current_tensor - prev_tensor).numpy()
            
            # 核心算法：计算多个指标来判断参数是否更新
            # 1. 平均绝对差异 (Mean Absolute Difference)
            mean_abs_diff = np.mean(np.abs(difference))
            
            # 2. 最大绝对差异 (Max Absolute Difference) - 检测是否有任何元素发生显著变化
            max_abs_diff = np.max(np.abs(difference))
            
            # 4. 相对变化比例 (针对非零参数)
            # 避免除零错误
            prev_norm = np.linalg.norm(prev_tensor.numpy())
            if prev_norm > 1e-10:
                relative_change = np.linalg.norm(difference) / prev_norm
            else:
                relative_change = np.linalg.norm(difference)
            
            # 判断逻辑：使用多重条件
            # 条件1: 平均绝对差异超过阈值
            condition1 = mean_abs_diff > self.threshold
            
            # 条件2: 最大差异超过阈值的10倍（检测是否有局部显著更新）
            condition2 = max_abs_diff > (self.threshold * 10)
            
            # 条件3: 相对变化超过0.001% (0.00001)
            # 这个条件对量化模型特别重要，因为量化后权重变化可能很小
            condition3 = relative_change > 0.00001
            
            # 最终判断：满足任一条件即认为参数已更新
            # 对于量化/冻结的参数，这些指标都应该接近0
            if condition1 or condition2 or condition3:
                modified_param_list.append((key, current_tensor.numel()))
            else:
                steady_param_list.append((key, current_tensor.numel()))
        
        return modified_param_list, steady_param_list
    
    def _process_queue(self):
        """
        后台线程处理队列中的参数快照
        对比当前参数与前一个快照，检测哪些参数被更新，哪些被冻结
        """
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # 检测参数更新情况（第一次调用时返回空的changed列表）
            is_first_call = (self.prev_item is None)
            changed, steady = self.updated_param_detection(item)
            
            # 更新prev_item供下次比较
            # 注意：必须在detection之后更新，否则会一直和自己比较
            epoch, step, tensor_dict = item
            self.prev_item = item
            
            # 跳过第一次调用（没有对比基准）
            if is_first_call:
                continue
            
            # 计算统计信息
            change_num, steady_num = len(changed), len(steady)
            total_params = change_num + steady_num
            
            # 安全地解包（处理空列表情况）
            if changed:
                changed_name_list, changed_params_num_list = zip(*changed)
            else:
                changed_name_list, changed_params_num_list = [], []
            
            if steady:
                unchanged_name_list, unchanged_params_num_list = zip(*steady)
            else:
                unchanged_name_list, unchanged_params_num_list = [], []
            
            # 避免除零错误
            if total_params > 0:
                steady_percentage = steady_num / total_params
                changed_percentage = change_num / total_params
            else:
                steady_percentage = 0.0
                changed_percentage = 0.0
            
            # 计算实际参数数量统计
            changed_params_count = sum(changed_params_num_list) if changed_params_num_list else 0
            unchanged_params_count = sum(unchanged_params_num_list) if unchanged_params_num_list else 0
            total_params_count = changed_params_count + unchanged_params_count
            
            # 构建结果字典
            batch_dict = {
                'epoch': epoch,
                'step': step,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_changed': change_num,
                'num_steady': steady_num,
                'total_params': total_params,
                'steady_percentage': f"{steady_percentage:.4f}",
                'changed_percentage': f"{changed_percentage:.4f}",
                'changed_params_nums': changed_params_count,
                'changed_params_percentage': f'{changed_params_count / total_params_count:.4f}' if total_params_count > 0 else '0.0000',
                'unchanged_params_nums': unchanged_params_count,  # 修正拼写错误
                'total_params_nums': total_params_count,
                'threshold_used': f"{self.threshold:.2e}",
                'changed_params': changed[:10] if len(changed) <= 10 else changed[:10] + [f"... and {len(changed)-10} more"],
                'steady_params': steady[:10] if len(steady) <= 10 else steady[:10] + [f"... and {len(steady)-10} more"]
            }
            
            # 保存到文件
            wpath = os.path.join(self.write_dir, f'param_analysis_{datetime.now().strftime("%m-%d")}.jsonl')
            with open(wpath, 'a', encoding='utf-8') as file:
                # 使用JSONL格式（每行一个JSON对象），方便追加和读取
                json_str = json.dumps(batch_dict, ensure_ascii=False, indent=2)
                file.write(json_str)
                file.write('\n')
            
            # 打印关键信息到控制台
            print(f"[Param Monitor] Epoch {epoch}, Step {step}: "
                  f"Changed={change_num}/{total_params} ({changed_percentage:.2%}), "
                  f"Frozen={steady_num}/{total_params} ({steady_percentage:.2%})")
        

    def end_process(self):
        print("[System] Stopping server")
        self.stop_event.set()
        self.worker_thread.join()
        print("[System] Server Terminated.")


# 全局实例 - 延迟初始化
_instance: Optional[AsyncSnapshotCapture] = None

def get_snapshot_saver(learning_rate: float = 2e-5, save_dir: str = "./outputs/params_update_check") -> AsyncSnapshotCapture:
    """
    获取参数监控器的全局实例（单例模式）
    
    Args:
        learning_rate: 学习率，用于设置检测阈值
        save_dir: 保存目录
    
    Returns:
        AsyncSnapshotCapture实例
    """
    global _instance
    if _instance is None:
        _instance = AsyncSnapshotCapture(save_dir=save_dir, learning_rate=learning_rate)
    return _instance


# ============ 使用示例 ============
# 在训练脚本中的使用方法：
# 
# 1. 在训练开始前初始化：
#    from BackPG_Test import get_snapshot_saver
#    snapshot_saver = get_snapshot_saver(learning_rate=2e-5)
# 
# 2. 在训练循环中定期保存参数快照（例如每100步）：
#    if step % 100 == 0:
#        param_dict = {name: param.data for name, param in model.named_parameters()}
#        snapshot_saver.save_snapshot(epoch=epoch, step=step, tensor_dict=param_dict)
# 
# 3. 训练结束时关闭：
#    snapshot_saver.end_process()
#
# 注意事项：
# - 第一次调用时没有对比基准，所有参数都会被标记为"steady"
# - 建议监控的参数包括所有可训练参数（requires_grad=True）
# - 对于量化模型，特别关注冻结层的参数是否真的没有变化
# - 输出文件为JSONL格式，每行一个JSON对象，便于流式读取和分析
