import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader_llama3 import sample_few_shot_blocks, sample_semi_supervised, EMOTION_MAP
from ZGeneration.config_gen import GenTrainingConfig

# Ensure pickle can load the gen data
import pickle

def load_gen_data(data_dir):
    cache_file = f"{data_dir}/processed_gen_data.pkl"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Generation data not found: {cache_file}")
    
    print(f"Loading Gen Data: {cache_file}")
    with open(cache_file, "rb") as f:
        data, max_len = pickle.load(f)
    print(f"Data Loaded: {len(data['input_text'])} samples.")
    return data, max_len

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
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
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
        
        input_data = self.data[self.prompt_key][real_idx] # Can be String or List[Dict]
        target_text = self.data[self.target_key][real_idx]
        
        # --- Handle New List[Dict] Format vs Old String Format ---
        if isinstance(input_data, list):
            # Format: List of messages [{'role':..., 'content':...}, ...]
            # Last message is the Assistant Target (Training Target)
            
            # 1. Full Sequence (Input + Target) for Training
            # apply_chat_template handles concatenation and special tokens
            full_input_str = self.tokenizer.apply_chat_template(input_data, tokenize=False)
            full_tokens = self.tokenizer(full_input_str, add_special_tokens=False, return_tensors='pt')
            full_input_ids = full_tokens['input_ids'].squeeze(0)
            
            # 2. Prompt Sequence (Input Only) for Inference/Masking
            # Remove the last message (Assistant Target) to get the prompt
            prompt_data = input_data[:-1]
            prompt_str = self.tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
            prompt_tokens = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors='pt')
            prompt_ids = prompt_tokens['input_ids'].squeeze(0)
            
            # 3. Create Labels (Mask Prompt with -100)
            # We want to train only on the new turn (Target)
            # Full sequence is [Prompt Tokens] + [Target Tokens] + [EOS?]
            # Actually, apply_chat_template(full) creates the whole string.
            # prompt_ids length tells us where to start computing loss.
            
            # Note: Tokenization of "Part A" + "Part B" is not always equal to Tokenization("Part A" + "Part B") due to spacing/merges.
            # But with Chat Templates, usually structure is clear. 
            # safe matching: Mask the first len(prompt_ids)
            
            labels = full_input_ids.clone()
            
            # Ensure prompt length doesn't exceed full length (sanity check)
            if len(prompt_ids) < len(full_input_ids):
                labels[:len(prompt_ids)] = -100
                # Mask potential trailing artifacts (EOS/Newline) if they cause mismatch with target_text
                # User observation: last 2 tokens are often <end_token>\n which don't match target_text
                # Note: Masking EOS might affect stopping behavior, but masking \n is safe.
                # However, to strictly follow the alignment observation:
                if len(labels) > 2:
                     labels[-1:] = -100 # Mask the very last 2 tokens (likely \n and EOS duplicate)
            else:
                # Fallback or weird edge case where prompt == full (no target?)
                labels[:] = -100
        else:
            raise ValueError("Input not follow process standard, check/refer to preload_gen_data.py")
        
        # --- Handle Padding / Truncation ---
        # Common logic for both paths once we have full_input_ids, labels, prompt_ids
        
        # 1. Pad Prompt (prompt_ids) to max_seq_len 
        # CRITICAL: Use LEFT PADDING for generation prompts. 
        # This ensures that generation starts immediately after the real tokens.
        
        prompt_input_ids = prompt_ids.clone()
        if len(prompt_input_ids) > self.max_seq_len:
             # Truncate left (keep last context)
             prompt_input_ids = prompt_input_ids[-self.max_seq_len:]
        else:
             pad_len_p = self.max_seq_len - len(prompt_input_ids)
             pad_ids_p = torch.full((pad_len_p,), self.tokenizer.pad_token_id, dtype=torch.long)
             # Left Padding: [PAD, ..., PAD, TOKENS]
             prompt_input_ids = torch.cat([pad_ids_p, prompt_input_ids], dim=0)
             
        prompt_attention_mask = (prompt_input_ids != self.tokenizer.pad_token_id).long()
        
        # 2. Pad/Truncate Full Input (For Training)
        # Strategy: Ensure Target is preserved as much as possible.
        # Training usually uses Right Padding with masked labels.
        if len(full_input_ids) > self.max_seq_len:
             # Truncate from left, assuming Target is at the end.
             full_input_ids = full_input_ids[-self.max_seq_len:]
             labels = labels[-self.max_seq_len:]
        else:
            # Pad to right
            pad_len = self.max_seq_len - len(full_input_ids)
            pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            # Mask padded region in labels
            pad_labels = torch.full((pad_len,), -100, dtype=torch.long)
            
            full_input_ids = torch.cat([full_input_ids, pad_ids], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
            
        attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": prompt_input_ids, 
            "prompt_mask": prompt_attention_mask,
            "target_text": target_text,
            "ud_idx": self.data['ud_idx'][real_idx],
            "ld_idx": self.data['ld_idx'][real_idx]
        }

def gen_loader_warp(data, tokenizer, config: GenTrainingConfig):
    """
    Lightweight loader warp for Generation Task.
    Reuses sampling logic.
    """
    # 1. Create a temporary dataset to get block mapping
    # (Checking all indices)
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
    # Remove train blocks
    remaining = sorted(list(set(all_block_idx) - set(train_blocks)))
    
    val_blocks = []
    test_blocks = []
    
    if config.fast_train:
        # Scenario 1: Fast Train is True
        # "valid and test should all be setup as 10 percent"
        # We apply this 10% logic to the TOTAL available blocks if possible, 
        # or just 10% of remaining if that's what's meant. 
        # Given "fast train", we usually want a small subset. 
        # Let's interpret as: Use 10% of TOTAL for Val and 10% of TOTAL for Test.
        
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
             
        # Also limit Train blocks if user meant "fast train" implies small dataset?
        # Re-reading: "under fast-train is True... valid and test ... 10 percent"
        # It doesn't explicitly restrict Train size, but usually fast_train does.
        # Existing logic for Full Train+Fast Train sliced top 80%.
        # If FSL/SSL, Train is already small.
        # So we just ensure Val/Test are 10%.
        
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

    # 4. Create Datasets
    train_ds = GenerationDataset.from_block_indices(data, tokenizer, config.max_seq_length, train_blocks, config.prompt_key)
    val_ds = GenerationDataset.from_block_indices(data, tokenizer, config.max_seq_length, val_blocks, config.prompt_key)
    test_ds = GenerationDataset.from_block_indices(data, tokenizer, config.max_seq_length, test_blocks, config.prompt_key)
    
    raw_ds = (train_ds, val_ds, test_ds)
    
    # 5. Loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, test_loader, raw_ds

def get_gen_dataloader(tokenizer, config: GenTrainingConfig):
    data_dir = os.path.join(config.data_path, 'gen_task') # Assuming subdir
    data, max_len = load_gen_data(data_dir)
    # config.max_seq_length = max_len # Disable this overwrite to respect args
    print(f"Dataset Max Len (Chars estimated): {max_len}. Using Config Max Seq Len: {config.max_seq_length}")
    return gen_loader_warp(data, tokenizer, config)
