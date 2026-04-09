# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
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
from ZGeneration.train_gen_fast_LM import calculate_per_sample_ppl
import nltk
from nltk.translate.bleu_score import sentence_bleu
import argparse

from LM_Code.data_module import EmpathyDataset as EmpathyDataset_old
from LM_Code.data_module import IAMMDataCollator, EMOTION_MAP
from LM_Code.train_module import EmotionHead
from src_analysis.metrics_func import calc_distinct

class RecordTrainer(Trainer):
    def __init__(self, emo_head, *args, **kwargs):
        self.base_dir = kwargs.pop('base_dir', './')
        self.pt_name = kwargs.pop('pt_name', 'record_tensors')
        super().__init__(*args, **kwargs)
        self.emo_head = emo_head.to(self.args.device) # step 4
        self.loss_fct = nn.CrossEntropyLoss() # step 5
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        
        situation_ids = inputs.pop("situation_input_ids")
        situation_mask = inputs.pop("situation_attention_mask")
        emotion_labels = inputs.pop("emotion_label")
        
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False
        )
        
        loss = outputs.loss
        
        """
        sit_outputs = model(
            input_ids=situation_ids,
            attention_mask=situation_mask,
            output_hidden_states=True
        )
        
        last_hidden_states = sit_outputs.hidden_states[-1]
        device = last_hidden_states.device
        
        emo_logits = self.emo_head(last_hidden_states, attention_mask=situation_mask.to(device))
        sit_emo_loss = self.loss_fct(emo_logits, emotion_labels.to(device))
        """
        total_loss = loss # + sit_emo_loss step 2 in night
        
        if (self.state.global_step+1) % 100 == 0:  # 每 100 步保存一次
            last_hidden_states = outputs.hidden_states[-1].detach().cpu()
            input_ids = input_ids.detach().cpu()
            attention_mask = attention_mask.detach().cpu()
            labels = labels.detach().cpu()
            
            torch.save({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "last_hidden_states": last_hidden_states
            }, os.path.join(self.base_dir, f"{self.pt_name}_{self.state.global_step}.pt"))

        return (total_loss, outputs) if return_outputs else total_loss
    
    # step 3 in night, optimizer is not the case
    def create_optimizer(self):
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad] # + list(self.emo_head.parameters())
            from torch.optim import AdamW
            self.optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        return self.optimizer
    

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
def run_training(train_context: np.ndarray, train_target: np.ndarray,\
    train_sit, train_emo, \
    SYSTEM_PROMPT: str, SEED: int, run_dir: str) -> None:
    global lr
    print("Building training dataset ...")
    train_dataset = EmpathyDataset_old(
        train_context, train_target, \
        train_sit, train_emo, \
        tokenizer, SYSTEM_PROMPT, max_length=512, sit_max_length=128
    )
    print(f"Training samples: {len(train_dataset)}\n")

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        seed=SEED,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,       # effective batch size = 16
        learning_rate=lr,
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

    data_collator = IAMMDataCollator(
        tokenizer=tokenizer,
        # model=model,
    )

    tensors_dir = os.path.join(run_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    emo_head = EmotionHead(
        hidden_size=model.config.hidden_size,
        num_emotions = len(EMOTION_MAP),
    )
    trainer = RecordTrainer(
        emo_head=emo_head,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        base_dir=tensors_dir,
        pt_name = "LM_training_step"
    )

    print("Starting LoRA fine-tuning ...")
    trainer.train()
    print("Fine-tuning complete.\n")

    model.save_pretrained(LORA_ADAPTER_PATH)
    tokenizer.save_pretrained(LORA_ADAPTER_PATH)
    emo_head.save(LORA_ADAPTER_PATH)
    print(f"LoRA adapter saved → {LORA_ADAPTER_PATH}\n")
    
    return emo_head



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
) -> tuple[str, float, float, float]:

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
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
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

    # ── Step 3: 根据 calculate_per_sample_ppl 计算 method 2 (sample_ppl) ───
    # 构建与训练对其的 labels（context 部分遮蔽为 -100）
    full_labels = full_ids.clone()
    full_labels[0, :context_len] = -100

    # calculate_per_sample_ppl 期待输入的 logits 和 labels 的形状均为 [Batch, Seq...] 
    sample_ppl_list = calculate_per_sample_ppl(logits, full_labels)
    sample_ppl = sample_ppl_list[0]

    return generated_response, loss.item(), ppl, sample_ppl


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

def compute_bleu(pred_t, ref_t):
    if not pred_t:
        pred_toks = []
    else:
        pred_toks = nltk.word_tokenize(pred_t)
        
    if not ref_t:
        ref_toks = []
    else:
        ref_toks = nltk.word_tokenize(ref_t)
    
    # Safety check for empty references or hypotheses
    if len(ref_toks) == 0 or len(pred_toks) == 0:
        b1, b2 = 0.0, 0.0
    else:
        # Trying Method 7 as it interpolates methods 4 and 5 (length smoothing + average counts)
        b1 = sentence_bleu([ref_toks], pred_toks, weights=(1, 0, 0, 0), )
        b2 = sentence_bleu([ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0), )
    
    return b1, b2
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--ratio', type=float, default = 0.1, help = "dataset ratio")
    parser.add_argument('--model_name', type=str, default = 'llama-3.2-1B')
    parser.add_argument('--new_model_train', action='store_true', help='Whether to train a new model even if an adapter already exists.')
    parser.add_argument('--task1', type=str, default='new_test', help='Task name for logging and saving purposes.')
    
    args = parser.parse_args()

    fname = './qwen11.txt'
    lr, ratio = 2e-5, args.ratio
    model_name = args.model_name
    new_model_train = True
    print(f"[ratio] the ratio be {args.ratio}")
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
    root_data = 'data/ED'
    train_context = np.load(os.path.join(root_data, 'sys_dialog_texts.train.npy'),  allow_pickle=True)
    train_target  = np.load(os.path.join(root_data, 'sys_target_texts.train.npy'),  allow_pickle=True)
    
    train_sit     = np.load(os.path.join(root_data, 'sys_situation_texts.train.npy'), allow_pickle=True)
    train_emo     = np.load(os.path.join(root_data, 'sys_emotion_texts.train.npy'), allow_pickle=True)

    test_context = np.load(os.path.join(root_data, 'sys_dialog_texts.test.npy'), allow_pickle=True)
    test_target  = np.load(os.path.join(root_data, 'sys_target_texts.test.npy'), allow_pickle=True)
    test_sit     = np.load(os.path.join(root_data, 'sys_situation_texts.test.npy'), allow_pickle=True)
    test_emo     = np.load(os.path.join(root_data, 'sys_emotion_texts.test.npy'), allow_pickle=True)

    assert len(train_context) == len(train_target), "train split length mismatch"
    assert len(test_context)  == len(test_target),  "test split length mismatch"



    # ── 获取 10% 的 训练数据 用于 微调 ──────────────────────────────────────────────
    indices = np.random.choice(len(train_context), size=int(len(train_context) * ratio), replace=False)
    train_context = train_context[indices]
    train_target  = train_target[indices]
    train_sit = train_sit[indices]
    train_emo = train_emo[indices]

    # ─── 加载模型 & Tokenizer ─────────────────────────────────────────────────────
    print(f"Loading model ...")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=f"../../LLModel/{model_name}",
        cache_dir=f'./{model_name}/',
        force_download=False,
    )

    # ← 必须加这几行
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=f"../../LLModel/{model_name}",
        torch_dtype=torch.float16,   # 显存不足可换 torch.bfloat16
        device_map="auto",           # 自动分配到 GPU / CPU
        cache_dir=f'./{model_name}/',
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
    run_dir = f"./LM_{model_name}/{model_name}_{args.task1}_{lr}_{int(ratio*100)}"
    os.makedirs(run_dir, exist_ok=True)

    LORA_ADAPTER_PATH = os.path.join(run_dir, "final_adapter")

    if not os.path.exists(LORA_ADAPTER_PATH) or new_model_train:
        emo_head = run_training(train_context, train_target, train_sit, train_emo, SYSTEM_PROMPT, SEED, run_dir)
    else:
        print(f"Found existing LoRA adapter at '{LORA_ADAPTER_PATH}', skipping training.\n")
        model.load_adapter(LORA_ADAPTER_PATH, adapter_name="default")    # ← 加载保存的 adapter
        model.set_adapter("default")              # ← 切换到 default adapter


    # ── 推理 & PPL / BLEU 计算 ────────────────────────────────────────────────
    ppl_total   = 0.0
    sample_ppl_total = 0.0
    bleu1_total = 0.0
    bleu2_total = 0.0
    bleu3_total = 0.0
    bleu4_total = 0.0
    qshbleu1, qshbleu2 = 0.0, 0.0

    all_results = []
    all_pred_tokens_corpus = []

    for i in range(len(test_context)):

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for j in range(len(test_context[i])):
            role = "user" if j % 2 == 0 else "assistant"
            history.append({"role": role, "content": test_context[i][j]})

        reference = test_target[i]

        # ── PPL 计算 ──────────────────────────────────────────────────────────
        generated, loss, ppl, sample_ppl = multi_turn_chat_with_ppl(
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
        qshb1, qshb2 = compute_bleu(generated, reference)

        bleu1_total += bleu1
        bleu2_total += bleu2
        bleu3_total += bleu3
        bleu4_total += bleu4
        qshbleu1 += qshb1
        qshbleu2 += qshb2
        ppl_total += ppl
        sample_ppl_total += sample_ppl

        pred_tokens = generated.strip().split()
        all_pred_tokens_corpus.extend(pred_tokens)

        all_results.append({
            "id": i,
            "history": history,
            "reference": reference,
            "generated": generated,
            "metrics": {
                "ppl": ppl,
                "sample_ppl": sample_ppl,
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4,
                # "my_bleu1": qshb1,
                # "my_bleu2": qshb2,
            }
        })

    corpus_dist_1, corpus_dist_2 = calc_distinct([res["generated"] for res in all_results], tokenizer)

    print(f"Average PPL: {ppl_total / len(test_context):.4f}")
    print(f"Average Sample PPL: {sample_ppl_total / len(test_context):.4f}")
    print(f"Average BLEU-1: {bleu1_total / len(test_context):.4f}")
    print(f"Average BLEU-2: {bleu2_total / len(test_context):.4f}")
    print(f"Average My BLEU-1: {qshbleu1 / len(test_context):.4f}")
    print(f"Average My BLEU-2: {qshbleu2 / len(test_context):.4f}")
    print(f"Corpus Dist-1: {corpus_dist_1:.4f}")
    print(f"Corpus Dist-2: {corpus_dist_2:.4f}")

    # 保存 JSONL 结果并包含全部级别的 dist1/2 指标
    for item in all_results:
        item["metrics"]["dist1_corpus"] = corpus_dist_1
        item["metrics"]["dist2_corpus"] = corpus_dist_2
    
    output_jsonl_path = os.path.join(run_dir, f"eval_results_{lr}_{int(ratio*100)}.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Evaluation results saved to {output_jsonl_path}")