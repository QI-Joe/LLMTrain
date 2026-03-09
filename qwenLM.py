# -*- coding: utf-8 -*-

import os
import sys
# from Example_Output_Txt_Function import Logging_and_Data_Initialization

import math
import random
import torch
from pprint import *
import torch.nn as nn
import numpy as np
from collections import Counter
#from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType


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
            context_len = len(tokenizer(context_text)["input_ids"])

            # 截断超长序列
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            # context 已超出 max_length → target 部分为空 → 跳过
            if context_len >= len(full_ids):
                skipped += 1
                continue

            # context 部分用 -100 遮蔽，只在 target 部分计算 loss
            labels = [-100] * context_len + full_ids[context_len:]

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long), # 模型 输入 & 输出 的 文本, 即 对话历史 + 标准回复;
                "labels":    torch.tensor(labels,   dtype=torch.long), # 模型 输出 的 文本, 即 标准回复;
            })

        if skipped:
            print(f"[Dataset] Skipped {skipped} samples (context >= max_length).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



# ─── 训练函数 ─────────────────────────────────────────────────────────────────
def run_training(train_context: np.ndarray, train_target: np.ndarray, SYSTEM_PROMPT: str, SEED: int) -> None:

    print("Building training dataset ...")
    train_dataset = EmpathyDataset(
        train_context, train_target, tokenizer, SYSTEM_PROMPT, max_length=512
    )
    print(f"Training samples: {len(train_dataset)}\n")

    training_args = TrainingArguments(
        output_dir="./Qwen3-8B/qwen3_lora_empathy2",
        seed=SEED,
        num_train_epochs=7,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,       # effective batch size = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,         # 显著省显存，8B 模型很有必要
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",                    # 不需要 wandb 等
        dataloader_num_workers=0,
        remove_unused_columns=False,         # 必须关闭，否则 Trainer 会删掉自定义字段
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning ...")
    trainer.train()
    print("Fine-tuning complete.\n")

    model.save_pretrained(LORA_ADAPTER_PATH)
    tokenizer.save_pretrained(LORA_ADAPTER_PATH)
    print(f"LoRA adapter saved → {LORA_ADAPTER_PATH}\n")



# ─── 推理 & PPL 计算 ──────────────────────────────────────────────────────────
def multi_turn_chat_with_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    DEVICE: str,
    history: list[dict],
    reference_answer: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, float, float]:

    model.eval()

    # ── Step 1: 自由生成回复 ──────────────────────────────────────────────────
    context_text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(context_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens         = output_ids[0][inputs["input_ids"].shape[-1]:]
    generated_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Step 2: 计算标准回复的 PPL ────────────────────────────────────────────
    full_text = tokenizer.apply_chat_template(
        history + [{"role": "assistant", "content": reference_answer}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    full_ids    = tokenizer(full_text,    return_tensors="pt")["input_ids"].to(DEVICE)
    context_ids = tokenizer(context_text, return_tensors="pt")["input_ids"].to(DEVICE)
    context_len = context_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits          # [1, seq_len, vocab_size]

    shift_logits = logits[0, context_len - 1 : -1, :]   # 预测 target 各 token
    shift_labels = full_ids[0, context_len:]             # target token ids

    loss = nn.CrossEntropyLoss(reduction="mean")(shift_logits, shift_labels)
    ppl  = torch.exp(loss).item()

    return generated_response, loss, ppl


# ─── BLEU 计算（纯 Python，无需 nltk）────────────────────────────────────────
def _modified_precision(hypothesis: list, reference: list, n: int) -> float:
    """
    计算 n-gram 修正精度，并应用 method1 平滑（epsilon=0.1）：
    当 numerator==0 时加 0.1，避免短句 BLEU 归零。
    与 nltk SmoothingFunction().method1 行为完全一致。
    """
    hyp_ngrams = (
        Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        if len(hypothesis) >= n else Counter()
    )
    ref_ngrams = (
        Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        if len(reference) >= n else Counter()
    )

    numerator   = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
    denominator = max(1, sum(hyp_ngrams.values()))

    # method1 smoothing
    if numerator == 0:
        numerator = 0.1

    return numerator / denominator


def compute_sentence_bleu(hypothesis_str: str, reference_str: str) -> tuple[float, float, float, float]:
    """
    计算 Cumulative BLEU-1 ~ BLEU-4。
    以空格分词（英文对话场景）。
    与 nltk sentence_bleu(weights=(...), smoothing_function=SmoothingFunction().method1) 等价。
    """
    hyp = hypothesis_str.strip().split()
    ref = reference_str.strip().split()

    if not hyp:
        return 0.0, 0.0, 0.0, 0.0

    # Brevity Penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))

    # 各阶修正精度 p1 ~ p4
    p = [_modified_precision(hyp, ref, n) for n in range(1, 5)]

    bleu1 = bp * p[0]
    bleu2 = bp * math.exp(0.5  * math.log(p[0]) + 0.5  * math.log(p[1]))
    bleu3 = bp * math.exp(1/3  * math.log(p[0]) + 1/3  * math.log(p[1]) + 1/3  * math.log(p[2]))
    bleu4 = bp * math.exp(0.25 * math.log(p[0]) + 0.25 * math.log(p[1]) + 0.25 * math.log(p[2]) + 0.25 * math.log(p[3]))

    return bleu1, bleu2, bleu3, bleu4



if __name__ == "__main__":


    fname = './qwen11.txt'
    # original_stdout, original_stderr, logger, stderr_logger = Logging_and_Data_Initialization(fname)


    # ── 固定随机种子，保证实验可复现 ─────────────────────────────────────────
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)        # 多卡时也生效
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    SYSTEM_PROMPT = "You are the assistant trying to show your empathy to the user during the "\
                    "conversation. Please don't over reply to the user's message (i.e., no need to use so many sentences.). "\
                    "Reply to the user's message as naturally as possible."


    # ── 加载数据 ──────────────────────────────────────────────────────────────
    train_context = np.load(os.path.join('data', 'sys_dialog_texts.train.npy'),  allow_pickle=True)
    train_target  = np.load(os.path.join('data', 'sys_target_texts.train.npy'),  allow_pickle=True)
    test_context  = np.load(os.path.join('data', 'sys_dialog_texts.test.npy'),   allow_pickle=True)
    test_target   = np.load(os.path.join('data', 'sys_target_texts.test.npy'),   allow_pickle=True)

    assert len(train_context) == len(train_target), "train split length mismatch"
    assert len(test_context)  == len(test_target),  "test split length mismatch"



    # ── 获取 10% 的 训练数据 用于 微调 ──────────────────────────────────────────────
    indices = np.random.choice(len(train_context), size=int(len(train_context) * 0.1), replace=False)
    train_context = train_context[indices]
    train_target  = train_target[indices]



    # ─── 加载模型 & Tokenizer ─────────────────────────────────────────────────────
    print(f"Loading model ...")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        cache_dir='./Qwen3-8B/',
        force_download=False,
    )

    # ← 必须加这几行
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        torch_dtype=torch.float16,   # 显存不足可换 torch.bfloat16
        device_map="auto",           # 自动分配到 GPU / CPU
        cache_dir='./Qwen3-8B/',
        force_download=False,
    )



    # ─── LoRA 配置 ────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # rank；显存紧张可改为 8
        lora_alpha=32,                 # scaling = lora_alpha / r = 2
        target_modules=[               # Qwen3 的 Attention + FFN 投影层
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    # device_map="auto" + gradient_checkpointing 时必须加这行，否则梯度断流
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print()



    # ── 微调（adapter 已存在则跳过，方便重复实验）────────────────────────────
    LORA_ADAPTER_PATH = "./Qwen3-8B/qwen3_lora_empathy2/final_adapter"

    if not os.path.exists(LORA_ADAPTER_PATH):
        run_training(train_context, train_target, SYSTEM_PROMPT, SEED)
    else:
        print(f"Found existing LoRA adapter at '{LORA_ADAPTER_PATH}', skipping training.\n")
        model.load_adapter(LORA_ADAPTER_PATH, adapter_name="default")    # ← 加载保存的 adapter
        model.set_adapter("default")              # ← 切换到 default adapter


    # ── 推理 & PPL / BLEU 计算 ────────────────────────────────────────────────
    ppl_total   = 0.0
    bleu1_total = 0.0
    bleu2_total = 0.0
    bleu3_total = 0.0
    bleu4_total = 0.0

    for i in range(len(test_context)):

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for j in range(len(test_context[i])):
            role = "user" if j % 2 == 0 else "assistant"
            history.append({"role": role, "content": test_context[i][j]})

        reference = test_target[i]

        # ── PPL 计算 ──────────────────────────────────────────────────────────
        generated, loss, ppl = multi_turn_chat_with_ppl(
            model=model,
            tokenizer=tokenizer,
            DEVICE=DEVICE,
            history=history,
            reference_answer=reference,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        # ── BLEU 计算 ──────────────────────────────────────────────────────────
        bleu1, bleu2, bleu3, bleu4 = compute_sentence_bleu(generated, reference)

        bleu1_total += bleu1
        bleu2_total += bleu2
        bleu3_total += bleu3
        bleu4_total += bleu4
        ppl_total += ppl

        print(f"【第 {i + 1} 条数据】")
        print("对话历史：")
        pprint(history)
        print("标准回复：")
        print(reference)
        print()
        print(f"【模型生成回复】\n{generated}")
        print(f"【损失】{loss:.4f}")
        print(f"【标准回复 PPL】{ppl:.4f}")
        print(f"【BLEU-1】{bleu1:.4f}")
        print(f"【BLEU-2】{bleu2:.4f}")
        print(f"【BLEU-3】{bleu3:.4f}")
        print(f"【BLEU-4】{bleu4:.4f}")
        print()


    print(f"Average PPL: {ppl_total / len(test_context):.4f}")
    print(f"Average BLEU-1: {bleu1_total / len(test_context):.4f}")
    print(f"Average BLEU-2: {bleu2_total / len(test_context):.4f}")
    print(f"Average BLEU-3: {bleu3_total / len(test_context):.4f}")
    print(f"Average BLEU-4: {bleu4_total / len(test_context):.4f}")