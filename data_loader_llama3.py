"""
数据加载与采样 - Llama3 情感分类微调
对齐原 IAMM 数据格式，支持 Few-shot/Semi-supervised Learning
"""
import os
import pickle
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from config_llama3 import TrainingConfig
from transformers import AutoTokenizer


# 32个情感标签 (EmpatheticDialogues)
EMOTION_MAP = {
    "afraid": 0, "angry": 1, "annoyed": 2, "anticipating": 3, "anxious": 4,
    "apprehensive": 5, "ashamed": 6, "caring": 7, "confident": 8, "content": 9,
    "devastated": 10, "disappointed": 11, "disgusted": 12, "embarrassed": 13,
    "excited": 14, "faithful": 15, "furious": 16, "grateful": 17, "guilty": 18,
    "hopeful": 19, "impressed": 20, "jealous": 21, "joyful": 22, "lonely": 23,
    "nostalgic": 24, "prepared": 25, "proud": 26, "sad": 27, "sentimental": 28,
    "surprised": 29, "terrified": 30, "trusting": 31
}
MAX_CTX_LEN: int = None
MAX_SIT_LEN: int = None

def load_empathetic_data(data_dir):
    """
    加载处理过的EmpatheticDialogues数据 (对齐preload_data.py格式)
    """
    cache_file = f"{data_dir}/processed_data.pkl"
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"数据文件不存在: {cache_file}\n"
            f"请先运行 src/utils/data/loader.py 生成预处理数据"
        )
    
    print(f"加载数据: {cache_file}")
    with open(cache_file, "rb") as f:
        data, max_ws_len, max_wo_len = pickle.load(f)
    
    print(f"Overall data size: {len(data['emotion'])} 样本")
    
    return data, max_ws_len, max_wo_len

def sample_few_shot_blocks(data_dict, shots_per_class=16, seed=42):
    """
    Few-shot采样: 按对话块采样，确保对话完整性
    
    Args:
        data_dict: 数据字典 (包含 "ud_idx" 和 "emotion" 键)
        shots_per_class: 每个类别的block数
        seed: 随机种子
        
    Returns:
        sampled_block_indices: 采样的block索引列表
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 构建block到emotion的映射（每个block只有一个emotion）
    block_emotions = {}
    for idx, (ud_idx, emotion) in enumerate(zip(data_dict["ud_idx"], data_dict["emotion"])):
        if ud_idx not in block_emotions:
            block_emotions[ud_idx] = emotion
    
    # 按情感标签分组blocks
    emotion_to_blocks = defaultdict(list)
    for block_idx, emotion in block_emotions.items():
        emotion_to_blocks[emotion].append(block_idx)
    
    sampled_blocks = []
    for emotion, blocks in emotion_to_blocks.items():
        if len(blocks) <= shots_per_class:
            sampled_blocks.extend(blocks)
        else:
            # print("random sample activated")
            sampled = random.sample(blocks, shots_per_class)
            sampled_blocks.extend(sampled)
    
    print(f"Few-shot采样 (block级别): {len(sampled_blocks)} blocks ({shots_per_class} blocks/class)")
    return sorted(sampled_blocks)


def blocks_to_prompt_indices(data_dict, block_indices: List[int]):
    """
    将block索引转换为prompt索引
    
    Args:
        data_dict: 数据字典 (包含 "ud_idx" 键)
        block_indices: block索引列表
        
    Returns:
        prompt_indices: 对应的prompt索引列表（保持顺序）
    """
    block_set = set(block_indices)
    prompt_indices = []
    
    for idx, ud_idx in enumerate(data_dict["ud_idx"]):
        if ud_idx in block_set:
            prompt_indices.append(idx)
    
    return prompt_indices


def sample_semi_supervised(block_array, labeled_ratio=0.1, seed=42):
    """
    Semi-supervised采样: 随机选择部分blocks作为labeled数据
    
    Args:
        block_array: block索引数组
        labeled_ratio: 标注数据比例 (0.1 = 10%)
        seed: 随机种子
        
    Returns:
        labeled_indices: 有标签的block索引列表（已排序，保持顺序）
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建副本避免修改原数组
    block_list = list(block_array.copy())
    random.shuffle(block_list)
    
    split_point = int(len(block_list) * labeled_ratio)
    labeled_blocks = block_list[:split_point]
    
    # 排序以保持原始顺序（重要：防止打乱对话顺序）
    labeled_indices = sorted(labeled_blocks)
    
    print(f"Semi-supervised采样: {len(labeled_indices)} labeled blocks ({labeled_ratio*100:.1f}%)")
    
    return labeled_indices
    


class EmpatheticDataset(Dataset):
    """
    支持双模式的Dataset: Block模式（采样用）和Prompt模式（训练用）
    
    Block模式: 用于few-shot采样，保持对话完整性
    Prompt模式: 用于训练，按prompt级别加载数据
    """
    
    def __init__(self, data_dict: Dict[str, List[str]], tokenizer, max_seq_len: int, 
                 key: str = 'ws_prompt',
                 block_mode: bool = False, indices: Optional[List[int] | torch.Tensor]=None):
        """
        Args:
            data_dict: 数据字典 (必须包含 'ud_idx', 'ld_idx', 'emotion', key)
            tokenizer: Llama3 tokenizer
            max_seq_len: 最大序列长度
            key: 使用的prompt键 ('ws_prompt' 或 'wo_prompt')
            block_mode: True=block模式（采样用），False=prompt模式（训练用）
            indices: 使用的索引列表（block_mode决定是block索引还是block索引还是prompt索引）
        """
        self.data = data_dict
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.key = key
        self.block_mode = block_mode
        
        # 构建block索引映射: block_idx -> [prompt_indices]
        self._build_block_index()
        
        # 根据模式设置索引
        if block_mode:
            # Block模式: indices是block索引列表
            self.indices = indices if indices is not None else list(self.block_map.keys())
        else:
            # Prompt模式: indices是prompt索引列表
            self.indices = indices if indices is not None else list(range(len(data_dict["emotion"])))
    
    @classmethod
    def from_block_indices(cls, data_dict, tokenizer, max_seq_len, block_indices, key='ws_prompt'):
        """
        从block索引快速创建prompt模式的训练dataset（便捷方法）
        
        Args:
            data_dict: 数据字典
            tokenizer: tokenizer实例
            max_seq_len: 最大序列长度
            block_indices: block索引列表（来自few-shot采样）
            key: prompt键
            
        Returns:
            EmpatheticDataset: prompt模式的dataset实例
            
        Example:
            >>> sampled_blocks = sample_few_shot_blocks(data, shots_per_class=16)
            >>> train_dataset = EmpatheticDataset.from_block_indices(
            ...     data, tokenizer, max_ws, sampled_blocks
            ... )
        """
        prompt_indices = blocks_to_prompt_indices(data_dict, block_indices)
        return cls(data_dict, tokenizer, max_seq_len, key=key,
                   block_mode=False, indices=prompt_indices)
    
    @classmethod
    def from_prompt_indices(cls, data_dict, tokenizer, max_seq_len, prompt_indices, key='ws_prompt'):
        """
        从prompt索引直接创建dataset（便捷方法）
        
        Args:
            data_dict: 数据字典
            tokenizer: tokenizer实例
            max_seq_len: 最大序列长度
            prompt_indices: prompt索引列表
            key: prompt键
            
        Returns:
            EmpatheticDataset: prompt模式的dataset实例
        """
        return cls(data_dict, tokenizer, max_seq_len, key=key,
                   block_mode=False, indices=prompt_indices)
    
    def _build_block_index(self):
        """构建对话块索引映射: ud_idx -> [prompt_indices]"""
        self.block_map = defaultdict(list)
        for idx, ud_idx in enumerate(self.data['ud_idx']):
            self.block_map[ud_idx].append(idx)
        self.block_map = dict(self.block_map)  # 转为普通dict
        
        # 按照local_dialogue_idx排序确保顺序正确
        for block_idx in self.block_map:
            prompt_indices = self.block_map[block_idx]
            # 按local_dialogue_idx排序
            prompt_indices.sort(key=lambda i: self.data['ld_idx'][i])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.block_mode:
            # Block模式: 返回整个对话块
            return self._get_block(idx)
        else:
            # Prompt模式: 返回单条prompt
            return self._get_single(idx)
    
    def _get_single(self, idx):
        """Prompt模式: 返回单条数据"""
        real_idx = self.indices[idx]
        
        # 提取数据
        context = self.data[self.key][real_idx]
        emotion = self.data["emotion"][real_idx]
        
        # Tokenization
        encoding = self.tokenizer(
            context,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(EMOTION_MAP[emotion], dtype=torch.long),
            "emotion_text": emotion,
            "block_idx": self.data['ud_idx'][real_idx],  # 用于debug
            "local_turn_idx": self.data['ld_idx'][real_idx],      # 用于debug
            "num_turns": 1
        }
    
    def _get_block(self, idx):
        """Block模式: 返回整个对话块（多条prompts）"""
        block_idx = self.indices[idx]
        prompt_indices = self.block_map[block_idx]
        
        # 收集该block的所有数据
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # 获取block的emotion（所有turn共享同一个emotion）
        emotion = self.data["emotion"][prompt_indices[0]]
        
        for prompt_idx in prompt_indices:
            context = self.data[self.key][prompt_idx]
            
            encoding = self.tokenizer(
                context,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids_list.append(encoding["input_ids"].squeeze(0))
            attention_mask_list.append(encoding["attention_mask"].squeeze(0))
            labels_list.append(torch.tensor(EMOTION_MAP[emotion], dtype=torch.long))
        
        return {
            "input_ids": torch.stack(input_ids_list),          # [batch, seq_len]
            "attention_mask": torch.stack(attention_mask_list), # [batch, seq_len]
            "labels": torch.stack(labels_list),   # [batch]
            "emotion_text": emotion,
            "block_idx": block_idx,
            "local_turn_idx": -1,
            "num_turns": len(prompt_indices),
        }


def loader_warp(data: Dict[str, List[Optional[str|int|torch.Tensor]]], tokenizer, config, max_seq_len: int):
    """
    创建Train/Val/Test DataLoaders，确保数据不泄漏且保持对话顺序
    
    Args:
        data: 数据字典
        tokenizer: tokenizer实例
        config: 配置对象
        max_seq_len: 最大序列长度
        
    Returns:
        train_loader, val_loader, test_loader
        
    注意：
        - 所有block索引都会排序，确保对话顺序不被打乱
        - Train/Val/Test的blocks互不重叠，防止数据泄漏
        - DataLoader设置shuffle=False保持顺序
    """
    # 1. 创建block模式dataset用于获取block_map
    block_dataset = EmpatheticDataset(
        data_dict=data, tokenizer=tokenizer,
        max_seq_len=max_seq_len, key=config.prompt_key, block_mode=True
    )
    
    # 2. 获取所有block索引
    all_block_idx = np.array(list(block_dataset.block_map.keys()))
    
    # 3. 根据模式采样训练集blocks
    if config.few_shot:
        train_blocks = sample_few_shot_blocks(data, config.shots_per_class)
    elif config.semi_supervised:
        train_blocks = sample_semi_supervised(all_block_idx, config.semi_ratio)
    else:
        # 全量训练：80% train, 10% val, 10% test
        np.random.seed(42)
        shuffled_blocks = all_block_idx.copy()
        np.random.shuffle(shuffled_blocks)
        
        train_split = int(len(shuffled_blocks) * 0.8)
        val_split = int(len(shuffled_blocks) * 0.9)
        
        train_blocks = sorted(shuffled_blocks[:train_split])
        val_blocks = sorted(shuffled_blocks[train_split:val_split])
        test_blocks = sorted(shuffled_blocks[val_split:])
    
    # 4. Few-shot/Semi-supervised模式：从剩余blocks分配val和test
    if config.few_shot or config.semi_supervised:
        # 计算剩余的blocks（未用于训练的）
        train_blocks_set = set(train_blocks)
        remaining_blocks = sorted([b for b in all_block_idx if b not in train_blocks_set])
        
        # 可选：fast_train模式只使用20%的val/test数据
        if config.fast_train and len(remaining_blocks) > 100:
            sample_size = max(100, int(len(remaining_blocks) * 0.2))
            np.random.seed(42)
            remaining_blocks = sorted(np.random.choice(remaining_blocks, sample_size, replace=False))
        
        # 平均分配给val和test
        mid_point = len(remaining_blocks) // 2
        val_blocks = remaining_blocks[:mid_point]
        test_blocks = remaining_blocks[mid_point:]
    
    # 5. 验证数据不泄漏
    train_set = set(train_blocks)
    val_set = set(val_blocks)
    test_set = set(test_blocks)
    
    assert len(train_set & val_set) == 0, "数据泄漏：train和val有重叠！"
    assert len(train_set & test_set) == 0, "数据泄漏：train和test有重叠！"
    assert len(val_set & test_set) == 0, "数据泄漏：val和test有重叠！"
    
    print(f"\n数据集划分:")
    print(f"  Train: {len(train_blocks)} blocks")
    print(f"  Val:   {len(val_blocks)} blocks")
    print(f"  Test:  {len(test_blocks)} blocks")
    
    # 6. 创建datasets（所有blocks已排序，保持对话顺序）
    train_dataset = EmpatheticDataset.from_block_indices(
        data, tokenizer, max_seq_len, train_blocks, key=config.prompt_key
    )
    val_dataset = EmpatheticDataset.from_block_indices(
        data, tokenizer, max_seq_len, val_blocks, key=config.prompt_key
    )
    test_dataset = EmpatheticDataset.from_block_indices(
        data, tokenizer, max_seq_len, test_blocks, key=config.prompt_key
    )
    
    print(f"  Train prompts: {len(train_dataset)}")
    print(f"  Val prompts:   {len(val_dataset)}")
    print(f"  Test prompts:  {len(test_dataset)}\n")
    
    # 7. 创建DataLoaders（shuffle=False保持对话顺序）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,  # 重要：不shuffle
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader  # 修复：返回创建的DataLoader train_loader, val_loader, test_loader


def get_dataloader(tokenizer, config: TrainingConfig):
    data_dir = config.data_path
    data, max_ws_len, max_wo_len = load_empathetic_data(data_dir)

    # DEBUG CHECK
    try:
        idx_check = 78
        print(f"\n[DEBUG] Checking Index {idx_check} in loaded data:")
        print(f"[DEBUG] Prompt: {data['ws_prompt'][idx_check][:50]}...")
        print(f"[DEBUG] Emotion: {data['emotion'][idx_check]}")
        print(f"[DEBUG] EMOTION_MAP['caring']: {EMOTION_MAP.get('caring')}")
        print(f"[DEBUG] EMOTION_MAP['furious']: {EMOTION_MAP.get('furious')}")
    except Exception as e:
        print(f"[DEBUG] Check failed: {e}")

    max_seq_len: int = max_wo_len
    if config.prompt_key == 'ws_prompt':
        max_seq_len = max_ws_len
    
    return loader_warp(data=data, tokenizer=tokenizer, config=config, max_seq_len=max_seq_len)
    

if __name__ == "__main__":
    """测试数据加载和顺序验证"""
    print("="*70)
    print("测试双模式数据加载 + get_dataloaders")
    print("="*70)
    
    # 加载数据
    data_dir = "./data/emotion_cls"
    data, max_ws_len, max_wo_len = load_empathetic_data(data_dir)
    
    # 加载tokenizer
    tokenPath = os.path.expanduser("~/Documents/LLModel/Llama-3.3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenPath)
    
    # 创建测试配置
    config = TrainingConfig()
    config.few_shot = True
    config.shots_per_class = 2
    config.batch_size = 4
    config.num_workers = 0
    config.fast_train = True
    
    # ===== 测试 get_dataloaders =====
    print("\n" + "="*70)
    print("测试 get_dataloaders (Few-shot模式)")
    print("="*70)
    
    train_loader, val_loader, test_loader = loader_warp(
        data, tokenizer, config, max_ws_len
    )
    
    # ===== 验证数据顺序 =====
    print("\n" + "="*70)
    print("验证对话顺序保持")
    print("="*70)
    
    batch_count = 0
    prev_dialogue_idx = -1
    prev_turn_idx = -1
    order_violations = 0
    
    for batch in train_loader:
        dialogue_ids = batch['block_idx'].tolist()
        turn_ids = batch['local_turn_idx'].tolist()
        
        for did, tid in zip(dialogue_ids, turn_ids):
            if did == prev_dialogue_idx:
                # 同一对话，turn应该递增
                if tid <= prev_turn_idx:
                    order_violations += 1
                    print(f"  ⚠ 顺序错误: dialogue {did}, turn {prev_turn_idx} -> {tid}")
            
            prev_dialogue_idx = did
            prev_turn_idx = tid
        
        batch_count += 1
        if batch_count >= 5:  # 只检查前5个batches
            break
    
    if order_violations == 0:
        print("  ✓ 前5个batches对话顺序正确！")
    else:
        print(f"  ✗ 发现 {order_violations} 处顺序错误")
    
    # ===== 验证数据不泄漏 =====
    # print("\n" + "="*70)
    # print("验证数据不泄漏")
    # print("="*70)
    
    # train_dialogues = set()
    # val_dialogues = set()
    # test_dialogues = set()
    
    # for batch in train_loader:
    #     train_dialogues.update(batch['dialogue_idx'].tolist())
    
    # for batch in val_loader:
    #     val_dialogues.update(batch['dialogue_idx'].tolist())
    
    # for batch in test_loader:
    #     test_dialogues.update(batch['dialogue_idx'].tolist())
    
    # train_val_overlap = train_dialogues & val_dialogues
    # train_test_overlap = train_dialogues & test_dialogues
    # val_test_overlap = val_dialogues & test_dialogues
    
    # print(f"  Train dialogues: {len(train_dialogues)}")
    # print(f"  Val dialogues:   {len(val_dialogues)}")
    # print(f"  Test dialogues:  {len(test_dialogues)}")
    # print(f"  Train ∩ Val:     {len(train_val_overlap)}")
    # print(f"  Train ∩ Test:    {len(train_test_overlap)}")
    # print(f"  Val ∩ Test:      {len(val_test_overlap)}")
    
    # if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    #     print("  ✓ 无数据泄漏！")
    # else:
    #     print("  ✗ 发现数据泄漏！")
    
    # ===== 展示示例batch =====
    print("\n" + "="*70)
    print("示例Batch")
    print("="*70)
    
    batch = next(iter(train_loader))
    print(f"  Batch size: {batch['input_ids'].shape}")
    print(f"  Emotions: {batch['emotion_text']}")
    print(f"  Dialogue IDs: {batch['block_idx'].tolist()}")
    print(f"  Turn IDs: {batch['local_turn_idx'].tolist()}")
    
    print("\n✓ 所有测试完成!")
    

