# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
# from Example_Output_Txt_Function import Logging_and_Data_Initialization

import random
import torch
from pprint import *
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from LM_Code.data_module import EmotionClassificationDataset, EmotionClassificationCollator, EMOTION_MAP
from LM_Code.train_module import EmotionHead
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Invert EMOTION_MAP for mapping back to strings
INVERTED_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

class EmotionClassificationTrainer(Trainer):
    def __init__(self, emo_head, *args, **kwargs):
        self.base_dir = kwargs.pop('base_dir', './')
        self.pt_name = kwargs.pop('pt_name', 'record_tensors')
        super().__init__(*args, **kwargs)
        self.emo_head = emo_head.to(self.args.device) 
        self.loss_fct = nn.CrossEntropyLoss() 
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        emotion_labels = inputs.pop("emotion_label")
        
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        
        last_hidden_states = outputs.hidden_states[-1]
        device = last_hidden_states.device
        
        emo_logits = self.emo_head(last_hidden_states, attention_mask=attention_mask.to(device))
        loss = self.loss_fct(emo_logits, emotion_labels.to(device))

        outputs.emo_logits = emo_logits

        if (self.state.global_step+1) % 100 == 0:  
            last_hidden_states_cpu = last_hidden_states.detach().cpu()
            input_ids_cpu = input_ids.detach().cpu()
            attention_mask_cpu = attention_mask.detach().cpu()
            logits_cpu = emo_logits.detach().cpu()
            
            torch.save({
                "input_ids": input_ids_cpu,
                "attention_mask": attention_mask_cpu,
                "last_hidden_states": last_hidden_states_cpu,
                'logits': logits_cpu,
            }, os.path.join(self.base_dir, f"{self.pt_name}_{self.state.global_step}.pt"))

        return (loss, outputs) if return_outputs else loss
    
    def create_optimizer(self):
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad] + list(self.emo_head.parameters())
            from torch.optim import AdamW
            self.optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        return self.optimizer

# ─── 训练函数 ─────────────────────────────────────────────────────────────────
def run_training(train_context: np.ndarray, train_target: np.ndarray,
    train_sit, train_emo, 
    SYSTEM_PROMPT: str, SEED: int, run_dir: str) -> None:
    global lr
    print("Building training dataset ...")
    train_dataset = EmotionClassificationDataset(
        contexts=train_context, 
        targets=train_target, 
        situations=train_sit, 
        emotion_labels=train_emo, 
        tokenizer=tokenizer, 
        system_prompt=SYSTEM_PROMPT, 
        situation_flag=situation_flag,
        max_length=512
    )
    print(f"Training samples: {len(train_dataset)}\n")

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        seed=SEED,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,       
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,         
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",                    
        dataloader_num_workers=0,
        remove_unused_columns=False,         
    )

    data_collator = EmotionClassificationCollator(
        tokenizer=tokenizer,
    )

    tensors_dir = os.path.join(run_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    emo_head = EmotionHead(
        hidden_size=model.config.hidden_size,
        num_emotions = len(EMOTION_MAP),
    )
    trainer = EmotionClassificationTrainer(
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model for classification.")
    parser.add_argument('--ratio', type=float, default = 0.1, help = "dataset ratio")
    parser.add_argument('--new_model_train', action='store_true', help='Whether to train a new model even if an adapter already exists.')
    parser.add_argument('--situation_flag', action='store_true', help='Whether to include the situation in the context for training and evaluation.')
    parser.add_argument('--task1', type=str, default='evaluation', help='Task to perform: "training" or "evaluation".')
    parser.add_argument('--task2', type=str, default='kv_cache_have', help='Task to perform: "training" or "evaluation".')
    
    args = parser.parse_args()

    lr, ratio = 2e-5, args.ratio
    new_model_train = args.new_model_train
    situation_flag = args.situation_flag

    # ── 固定随机种子，保证实验可复现 ─────────────────────────────────────────
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SYSTEM_PROMPT = "You are an assistant predicting the user's emotion from the dialogue."
    print(f"{'-'*30} Starting Task: {args.task1} | Situation Included: {situation_flag} | Dataset Ratio: {int(ratio*100)}% {'-'*30}\n")

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
        pretrained_model_name_or_path="../../LLModel/llama3.1-8B-Instruct",
        cache_dir='./llama3-8B/',
        force_download=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="../../LLModel/llama3.1-8B-Instruct",
        torch_dtype=torch.float16,   
        device_map="auto",           
        cache_dir='./llama3-8B/',
        force_download=False,
    )

    # ─── LoRA 配置 ────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          
        lora_alpha=32,                 
        target_modules=[               
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print()

    # ── 微调（adapter 已存在则跳过，方便重复实验）────────────────────────────
    run_dir = f"./Llama3.1_8B_classification/llama3_{args.task1}_{lr}_{int(ratio*100)}"
    os.makedirs(run_dir, exist_ok=True)

    LORA_ADAPTER_PATH = os.path.join(run_dir, "final_adapter")

    if not os.path.exists(LORA_ADAPTER_PATH) or new_model_train:
        emo_head = run_training(train_context, train_target, train_sit, train_emo, SYSTEM_PROMPT, SEED, run_dir)
    else:
        print(f"Found existing LoRA adapter at '{LORA_ADAPTER_PATH}', skipping training.\n")
        model.load_adapter(LORA_ADAPTER_PATH, adapter_name="default")    
        model.set_adapter("default")              
        
        emo_head = EmotionHead(
            hidden_size=model.config.hidden_size,
            num_emotions=len(EMOTION_MAP),
        )
        emo_head.load_state_dict(torch.load(f"{LORA_ADAPTER_PATH}/emotion_head.pt", map_location=DEVICE))
        emo_head.to(DEVICE)

    # ── 推理 & 评估指标计算 ────────────────────────────────────────────────
    print("Building evaluation dataset ...")
    test_dataset = EmotionClassificationDataset(
        contexts=test_context, 
        targets=test_target, 
        situations=test_sit, 
        emotion_labels=test_emo, 
        tokenizer=tokenizer, 
        system_prompt=SYSTEM_PROMPT, 
        situation_flag=situation_flag,
        max_length=512
    )
    
    data_collator = EmotionClassificationCollator(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator, shuffle=False)

    model.eval()
    emo_head.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["emotion_label"].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            last_hidden_states = outputs.hidden_states[-1]
            emo_logits = emo_head(last_hidden_states, attention_mask=attention_mask)
            
            preds = torch.argmax(emo_logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.extend(emo_logits.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}\n")

    all_results = []
    for i in range(len(test_context)):
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for j, utt in enumerate(test_context[i]):
            role = "user" if j % 2 == 0 else "assistant"
            content = utt
            if j == 0 and situation_flag and test_sit[i]:
                content = f"Situation: {test_sit[i]}\n\nUser: {utt}"
            history.append({"role": role, "content": content})

        all_results.append({
            "id": i,
            "history": history,
            "reference": test_target[i],
            "true_emotion_idx": all_labels[i],
            "true_emotion_str": INVERTED_EMOTION_MAP.get(all_labels[i], "unknown"),
            "pred_emotion_idx": all_preds[i],
            "pred_emotion_str": INVERTED_EMOTION_MAP.get(all_preds[i], "unknown"),
            # "logits": all_logits[i],
            'metrics': {
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            },
            "situation": test_sit[i] if situation_flag else ""
        })

    output_jsonl_path = os.path.join(run_dir, f"eval_results_{lr}_{int(ratio*100)}_cls.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Evaluation results saved to {output_jsonl_path}")

