import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader_llama3 import sample_few_shot_blocks, sample_semi_supervised, EMOTION_MAP
from ZGeneration.config_gen import GenTrainingConfig

# Ensure pickle can load the gen data
import pickle
import glob
import re

def load_gen_data(data_dir):
    """
    Load processed generation data.
    Prefers the SlideWindow-named file (produced by the new preloader);
    falls back to the legacy 'processed_gen_data.pkl' if not found.
    Returns (data_dict, max_len, slide_windows) where slide_windows is an int
    inferred from the filename (or 0 if using the legacy file).
    """
    # 1. Look for SlideWindow-named file first
    sw_files = sorted(glob.glob(os.path.join(data_dir, 'processed_gen_SlideWindow_*.pkl')))
    if sw_files:
        cache_file = sw_files[-1]  # Use the latest if multiple exist
        m = re.search(r'SlideWindow_(\d+)\.pkl$', cache_file)
        slide_windows = int(m.group(1)) if m else 0
    else:
        # Fallback to legacy filename
        cache_file = os.path.join(data_dir, 'processed_gen_data.pkl')
        slide_windows = 0

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"No generation data pkl found in: {data_dir}\n"
            f"Expected 'processed_gen_SlideWindow_N.pkl' or 'processed_gen_data.pkl'"
        )

    print(f"Loading Gen Data: {cache_file}  (slide_windows={slide_windows})")
    with open(cache_file, 'rb') as f:
        data, max_len = pickle.load(f)
    print(f"Data Loaded: {len(data['input_text'])} samples.")
    return data, max_len, slide_windows

class GenerationDataset(Dataset):
    """
    Dataset for Llama 3 Generation Task.
    Concatenates Input + Target and masks labels for Input.
    """
    def __init__(self, data_dict: Dict, tokenizer, max_seq_len: int, 
                 indices: Optional[List[int]] = None,
                 prompt_key: str = 'input_text',
                 target_key: str = 'target_text'):
        
        self.data = data_dict
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.indices = indices if indices is not None else list(range(len(data_dict['emotion'])))
        self.prompt_key = prompt_key
        self.target_key = target_key
        
        # Build block mapping for sampling logic (reused from EmpatheticDataset concept)
        self._build_block_index()
        
    def _build_block_index(self):
        self.block_map = defaultdict(list)
        for idx in self.indices: # Only map available indices
            ud_idx = self.data['ud_idx'][idx]
            self.block_map[ud_idx].append(idx)
            
        # Sort based on local_dialogue_idx to maintain order in blocks
        for b_idx in self.block_map:
            self.block_map[b_idx].sort(key=lambda x: self.data['ld_idx'][x])

    @classmethod
    def from_block_indices(cls, data_dict, tokenizer, max_seq_len, block_indices, prompt_key='input_text'):
        """Create dataset from a list of block IDs (preserving order logic)"""
        # Collect all prompt indices belonging to these blocks
        # First build a global mapping or iterate?
        # To be efficient, we can iterate all data once or assume we have the mapping.
        # Let's do a quick pre-scan to map all data to blocks.
        
        # 1. Map all data to blocks first (Global map)
        global_block_map = defaultdict(list)
        for idx, ud_idx in enumerate(data_dict['ud_idx']):
            global_block_map[ud_idx].append(idx)
        
        # 2. Sort turns within block
        for b_idx in global_block_map:
            global_block_map[b_idx].sort(key=lambda x: data_dict['ld_idx'][x])
            
        # 3. Flatten requested blocks to sample indices
        final_indices = []
        for b_idx in block_indices:
            if b_idx in global_block_map:
                final_indices.extend(global_block_map[b_idx])
                
        return cls(data_dict, tokenizer, max_seq_len, indices=final_indices, prompt_key=prompt_key)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        input_data = self.data[self.prompt_key][real_idx]  # List[Dict] from new preloader

        if not isinstance(input_data, list):
            raise ValueError(
                f"Expected List[Dict] messages format from new preloader, "
                f"got {type(input_data)}. Re-run preload_gen_data_user_assist_turn.py."
            )

        # ── 1. Full conversation string ──────────────────────────────────────────────
        # apply_chat_template already injects <|begin_of_text|>, role headers, and EOS
        # so we MUST use add_special_tokens=False when calling the tokenizer directly.
        full_text = self.tokenizer.apply_chat_template(
            input_data, tokenize=False, add_generation_prompt=False,
            enable_thinking=False  # Disable Qwen3 thinking mode; ignored by Llama
        )

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,   # BOS/EOS already in the template string
            return_tensors=None,
        )
        input_ids      = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # ── 2. Build label mask ──────────────────────────────────────────────────────
        # Default: -100 everywhere (mask system prompt + all user turns).
        # Only assistant-response tokens are unmasked and used for loss.
        labels = [-100] * len(input_ids)

        for i, msg in enumerate(input_data):
            if msg["role"] != "assistant":
                continue  # system / user → stay masked

            # -- Start boundary: prefix up to (but not including) this assistant turn.
            #    add_generation_prompt=True appends the assistant role header token(s),
            #    so prefix_len points to the first token of the assistant's *content*.
            prefix_text = self.tokenizer.apply_chat_template(
                input_data[:i], tokenize=False, add_generation_prompt=True,
                enable_thinking=False  # Disable Qwen3 thinking mode; ignored by Llama
            )
            prefix_len = len(self.tokenizer(
                prefix_text, add_special_tokens=False, return_tensors=None
            )["input_ids"])

            # -- End boundary: full text up to and including this assistant turn.
            full_up_to = self.tokenizer.apply_chat_template(
                input_data[: i + 1], tokenize=False, add_generation_prompt=False,
                enable_thinking=False  # Disable Qwen3 thinking mode; ignored by Llama
            )
            end_len = len(self.tokenizer(
                full_up_to, add_special_tokens=False, return_tensors=None
            )["input_ids"])

            # Unmask assistant-content tokens (clamped to actual sequence length)
            for j in range(prefix_len, min(end_len, len(input_ids))):
                labels[j] = input_ids[j]

        # ── 3. Prompt-only string (kept for inference / custom evaluation) ───────────
        # System + history up to last user turn, with add_generation_prompt=True so
        # the model can directly continue with its generation at eval time.
        # NOTE: system prompt is intentionally included here — it frames the task
        # and is required context; it is simply *not* predicted (masked above).

        return {
            "input_ids":       torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask":  torch.tensor(attention_mask, dtype=torch.long),
            "labels":          torch.tensor(labels,         dtype=torch.long),
            "text":            input_data,  
            "emotion":         self.data["emotion"][real_idx],
            "ud_idx":          self.data["ud_idx"][real_idx],
            "ld_idx":          self.data["ld_idx"][real_idx],
        }

def gen_collate_fn(batch, pad_token_id: int = 0):
    """
    Custom collate function for GenerationDataset.

    Pads tensor fields (input_ids / attention_mask / labels) to the longest
    sequence in the batch; string / scalar metadata fields are collected into
    plain lists so they pass through the DataLoader without errors.

    Args:
        batch:         list of dicts returned by GenerationDataset.__getitem__
        pad_token_id:  token id used for right-padding input_ids & attention_mask
                       (labels padding is always -100 so it is ignored by the loss)
    """
    # Separate tensor fields from string/scalar metadata
    input_ids_list      = [item["input_ids"]      for item in batch]
    attention_mask_list = [item["attention_mask"]  for item in batch]
    labels_list         = [item["labels"]          for item in batch]
    text_list           = [item["text"]            for item in batch]

    max_len = max(t.size(0) for t in input_ids_list)

    def pad_tensor(tensor, pad_value, target_len):
        pad_size = target_len - tensor.size(0)
        if pad_size == 0:
            return tensor
        return torch.cat([tensor, torch.full((pad_size,), pad_value, dtype=tensor.dtype)])

    input_ids      = torch.stack([pad_tensor(t, pad_token_id, max_len) for t in input_ids_list])
    attention_mask = torch.stack([pad_tensor(t, 0,             max_len) for t in attention_mask_list])
    labels         = torch.stack([pad_tensor(t, -100,          max_len) for t in labels_list])

    return {
        "input_ids":      input_ids,       # (B, max_len)
        "attention_mask": attention_mask,  # (B, max_len)
        "labels":         labels,          # (B, max_len)  — -100 positions ignored by loss
        "text":    [item["text"] for item in batch],
        "emotion":        [item["emotion"]     for item in batch],
        "ud_idx":         [item["ud_idx"]      for item in batch],
        "ld_idx":         [item["ld_idx"]      for item in batch],
    }


def gen_loader_warp(data, tokenizer, config: GenTrainingConfig):
    """
    Lightweight loader warp for Generation Task.
    Reuses sampling logic from data_loader_llama3.
    tokenizer is stored in each dataset and used in __getitem__ for apply_chat_template.
    """
    # 1. Build block mapping using a tokenizer-free dataset
    #    (block_map only uses ud_idx/ld_idx — no tokenizer needed here)
    full_dataset = GenerationDataset(data, tokenizer, config.max_seq_length, prompt_key=config.prompt_key)
    all_block_idx = np.array(list(full_dataset.block_map.keys()))
    
    # 2. Train split logic (Reusing code flow from data_loader_llama3)
    train_blocks = []
    
    if config.few_shot:
        train_blocks = sample_few_shot_blocks(data, config.shots_per_class)
    elif config.semi_supervised:
        train_blocks = sample_semi_supervised(all_block_idx, config.semi_ratio)
    else:
        # Full train
        if config.fast_train:
            num_blocks = len(all_block_idx)
            # If Fast Train -> maybe smaller subset
            split1 = int(0.8 * num_blocks)
            train_blocks = list(all_block_idx[:split1])

    # 3. Val/Test Split
    remaining = sorted(list(set(all_block_idx) - set(train_blocks)))
    
    val_blocks = []
    test_blocks = []
    
    if config.fast_train:
        # Scenario 1: Fast Train is True
        # "valid and test should all be setup as 10 percent"
        num_total = len(all_block_idx)
        n_val = int(num_total * 0.1)
        n_test = int(num_total * 0.1)
        
        # Ensure we don't exceed remaining
        if n_val + n_test > len(remaining):
             # Fallback to splitting remaining evenly if not enough
             print("Warning: Fast Train 10% requirement exceeds remaining data. Splitting remaining 50/50.")
             mid = len(remaining) // 2
             val_blocks = remaining[:mid]
             test_blocks = remaining[mid:]
        else:
             val_blocks = remaining[:n_val]
             test_blocks = remaining[n_val : n_val + n_test]
        
    else:
        # Scenario 2: Fast Train is False
        if config.few_shot or config.semi_supervised:
            # "adjust number of validation and test in config"
            # Use val_ratio and test_ratio from config (default 0.1 / 0.1 or user set)
            
            num_total = len(all_block_idx)
            # Use getattr to be safe if config hasn't been reloaded/updated in memory yet
            r_val = getattr(config, 'val_ratio', 0.1) 
            r_test = getattr(config, 'test_ratio', 0.1)
            
            n_val = int(num_total * r_val)
            n_test = int(num_total * r_test)
            
            if n_val + n_test > len(remaining):
                 print(f"Warning: Configured Ratios ({r_val}, {r_test}) exceed remaining data. Splitting remaining 50/50.")
                 mid = len(remaining) // 2
                 val_blocks = remaining[:mid]
                 test_blocks = remaining[mid:]
            else:
                 val_blocks = remaining[:n_val]
                 test_blocks = remaining[n_val : n_val + n_test]
                 
        else:
            # Full Train (Standard)
            # "check how many remaining data left, then adjust rest of block to divde in even"
            mid = len(remaining) // 2
            val_blocks = remaining[:mid]
            test_blocks = remaining[mid:]

    print(f"Gen Data Split: Train {len(train_blocks)}, Val {len(val_blocks)}, Test {len(test_blocks)} blocks.")

    # 4. Create Datasets  (tokenizer is now the second positional arg — matches from_block_indices)
    train_ds = GenerationDataset.from_block_indices(data, tokenizer, config.max_seq_length, train_blocks, config.prompt_key)
    val_ds   = GenerationDataset.from_block_indices(data, tokenizer, config.max_seq_length, val_blocks,   config.prompt_key)
    test_ds  = GenerationDataset.from_block_indices(data, tokenizer, config.max_seq_length, test_blocks,  config.prompt_key)
    
    raw_ds = (train_ds, val_ds, test_ds)

    train_loader, val_loader, test_loader = None, None, None
    
    return train_loader, val_loader, test_loader, raw_ds

def get_gen_dataloader(tokenizer, config: GenTrainingConfig):
    data_dir = os.path.join(config.data_path, 'gen_task')
    data, max_len, slide_windows = load_gen_data(data_dir)
    print(f"Dataset Max Len (Chars estimated): {max_len}. "
          f"Using Config Max Seq Len: {config.max_seq_length}. "
          f"Slide windows: {slide_windows}.")
    return gen_loader_warp(data, tokenizer, config), slide_windows
