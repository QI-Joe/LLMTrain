import os
import sys
# from Example_Output_Txt_Function import Logging_and_Data_Initialization

import math
import random
import torch
from pprint import *
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset

SYSTEM_PROMPT = "You are the assistant trying to show your empathy to the user during the "\
                    "conversation. Please don't over reply to the user's message (i.e., no need to use so many sentences.). "\
                    "Reply to the user's message as naturally as possible."

# ─── Dataset ──────────────────────────────────────────────────────────────────
class EmpathyDataset(Dataset):
    """
    把 (context, target) 对格式化为完整对话序列。
    labels 中 context 部分全部置为 -100，只让模型在 target 部分计算 loss。
    """

    def __init__(
        self,
        contexts:      np.ndarray,
        targets:       np.ndarray,
        tokenizer:     AutoTokenizer,
        system_prompt: str,
        max_length:    int = 512,
    ):
        self.samples = []
        skipped = 0

        for context, target in zip(contexts, targets):

            # ── 构建对话 history ──────────────────────────────────────────────
            history = [{"role": "system", "content": system_prompt}]
            for j, utt in enumerate(context):
                role = "user" if j % 2 == 0 else "assistant"
                history.append({"role": role, "content": utt})

            # ── 生成文本字符串 ─────────────────────────────────────────────────
            # "context_text" 就是 模型 输入 的 文本, 即 单纯的 对话历史;
            #
            # context_text：末尾带 generation prompt，用来精确定位 context 边界
            context_text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
            # "full_text" 就是 模型 输入 & 输出 的 文本, 即 对话历史 + 标准回复;
            # full_text：context + 标准回复
            full_text = tokenizer.apply_chat_template(
                history + [{"role": "assistant", "content": target}],
                tokenize=False,
                add_generation_prompt=False,
            )

            # ── Tokenize ──────────────────────────────────────────────────────
            full_ids    = tokenizer(full_text)["input_ids"]
            ctx_id = tokenizer(context_text)["input_ids"]
            context_len = len(ctx_id)

            # 截断超长序列
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            # context 已超出 max_length → target 部分为空 → 跳过
            if context_len >= len(full_ids):
                skipped += 1
                continue
            
            if context_len > max_length:
                ctx_id = ctx_id[:max_length]
            # context 部分用 -100 遮蔽，只在 target 部分计算 loss
            labels = [-100] * context_len + full_ids[context_len:]

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long), # 模型 输入 & 输出 的 文本, 即 对话历史 + 标准回复;
                "labels":    torch.tensor(labels,   dtype=torch.long), # 模型 输出 的 文本, 即 标准回复;
                "prompt_ids": torch.tensor(ctx_id,    dtype=torch.long), # 模型 输入 的 文本, 即 单纯的 对话历史;
                "target_text": target
            })

        if skipped:
            print(f"[Dataset] Skipped {skipped} samples (context >= max_length).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataloader(tokenizer, system_prompt=SYSTEM_PROMPT):
    train_context = np.load(os.path.join('data/ED', 'sys_dialog_texts.train.npy'),  allow_pickle=True)
    train_target  = np.load(os.path.join('data/ED', 'sys_target_texts.train.npy'),  allow_pickle=True)
    test_context  = np.load(os.path.join('data/ED', 'sys_dialog_texts.test.npy'),   allow_pickle=True)
    test_target   = np.load(os.path.join('data/ED', 'sys_target_texts.test.npy'),   allow_pickle=True)
    
    train_context = train_context[: int(len(train_context)*0.2)]
    train_target  = train_target[: int(len(train_target)*0.2)]
    
    train_dataset = EmpathyDataset(train_context, train_target, tokenizer, system_prompt)
    test_dataset = EmpathyDataset(test_context,  test_target,  tokenizer, system_prompt)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(r'../../LLModel/llama3.1-8B-Instruct', trust_remote_code=True)
    train_dataset, test_dataset = get_dataloader(tokenizer)
    
    indices = np.random.choice(len(train_dataset), 5, replace=False)
    for idx in indices:
        sample = train_dataset[idx]
        print(f"Sample {idx}:")
        input_id2txt = tokenizer.batch_decode(sample["input_ids"], skip_special_tokens=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        sample["labels"][sample["labels"] == -100] = pad_id
        label_id2txt = tokenizer.batch_decode(sample["labels"], skip_special_tokens=True)
        print("Input text:", ''.join(input_id2txt))
        print("Label text:", ' '.join(label_id2txt))
        print("Sample Keys:", sample.keys())
        print("-" * 50, "\n")
        
    print(f"\nNow for testing dataset: {len(test_dataset)} samples. {'-'*50}\n")
    indices = np.random.choice(len(test_dataset), 5, replace=False)
    for idx in indices:
        sample = test_dataset[idx]
        print(f"Sample {idx}:")
        input_id2txt = tokenizer.batch_decode(sample["input_ids"], skip_special_tokens=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        sample["labels"][sample["labels"] == -100] = pad_id
        label_id2txt = tokenizer.batch_decode(sample["labels"], skip_special_tokens=True)
        print("Input text:", ''.join(input_id2txt))
        print("Label text:", ' '.join(label_id2txt))
        print("Sample Keys:", sample.keys())
        print("-" * 50, "\n")
    
