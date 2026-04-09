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
#from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from ZGeneration.train_gen_fast_LM import calculate_per_sample_ppl
import nltk
from nltk.translate.bleu_score import sentence_bleu
from LM_Code.data_module import EmpathyDatasetForPrediction as EmpathyDataset4Pred
from LM_Code.data_module import PredictionDataCollator, EMOTION_MAP
from torch.utils.data import DataLoader
from LM_Code.My_qwenLM import compute_bleu, compute_sentence_bleu
from src_analysis.metrics_func import calc_distinct
import argparse


def multi_turn_chat_with_ppl_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    DEVICE: str,
    batch: dict,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    model.eval()
    
    context_input_ids = batch["context_input_ids"].to(DEVICE)
    context_attention_mask = batch["context_attention_mask"].to(DEVICE)
    full_input_ids = batch["full_input_ids"].to(DEVICE)
    full_attention_mask = batch["full_attention_mask"].to(DEVICE)
    full_labels = batch["full_labels"].to(DEVICE)

    sample_indices = batch["sample_idx"]

    with torch.no_grad():
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        output_ids = model.generate(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    step_scores = output_ids.scores
    prompt_len = context_input_ids.shape[-1]
    new_tokens_tensor = output_ids.sequences[:, prompt_len:]
    generated_responses = tokenizer.batch_decode(new_tokens_tensor, skip_special_tokens=True)
    generated_seq_len = len(step_scores)
    
    batch_size = output_ids.sequences.shape[0]
    batch_results = []

    for batch_idx in range(batch_size):
        sample_idx = sample_indices[batch_idx]
        batch_predict_dict = {
            "idx": f"{sample_idx}",
            "generated": generated_responses[batch_idx],
            "top_10_tokens_pre_position": []
        }
        
        for step_idx in range(generated_seq_len):
            current_logits = step_scores[step_idx][batch_idx] 
            top_10_values, top_10_indices = torch.topk(current_logits, k=10)
            top_10_tokens = tokenizer.convert_ids_to_tokens(top_10_indices)
            
            if step_idx < new_tokens_tensor.shape[1]:
                chosen_token_id = new_tokens_tensor[batch_idx, step_idx].item()
                word = tokenizer.decode([chosen_token_id]) 
            else:
                word = ""
            
            if word == tokenizer.eos_token:
                continue
            top_10_tokens = [tok.replace("Ġ", "") for tok in top_10_tokens]
            batch_predict_dict["top_10_tokens_pre_position"].append({
                "word": word,
                "topk_when_generate": " ".join(top_10_tokens),
                "topk_scores": top_10_values.cpu().numpy().tolist()
            })
            
        batch_results.append(batch_predict_dict)

    # ── Step 2: Compute PPL using the stored full_input_ids directly
    with torch.no_grad():
        logits = model(
            input_ids=full_input_ids, 
            attention_mask=full_attention_mask
        ).logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = full_labels[..., 1:].contiguous()

    sample_ppl_list = calculate_per_sample_ppl(logits, full_labels)
    
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(batch_size, -1)
    
    mask = (shift_labels != -100).float()
    seq_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    ppl_list = torch.exp(seq_losses).cpu().numpy().tolist()
    loss_list = seq_losses.cpu().numpy().tolist()

    return generated_responses, loss_list, ppl_list, sample_ppl_list, batch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('--ratio', type=float, default = 0.1, help = "dataset ratio")
    parser.add_argument('--task1', type=str, default='evaluation', help='Task to perform: "training" or "evaluation".')
    parser.add_argument('--task2', type=str, default='kv_cache_have', help='Task to perform: "training" or "evaluation".')
    
    args = parser.parse_args()

    fname = './qwen11.txt'
    lr, ratio = 2e-5, args.ratio
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
        pretrained_model_name_or_path="../../LLModel/llama3.1-8B-Instruct",
        cache_dir='./llama3-8B/',
        force_download=False,
    )

    # ← 必须加这几行
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="../../LLModel/llama3.1-8B-Instruct",
        torch_dtype=torch.float16,   # 显存不足可换 torch.bfloat16
        device_map="auto",           # 自动分配到 GPU / CPU
        cache_dir='./llama3-8B/',
        force_download=False,
    )



    # ── 微调（adapter 已存在则跳过，方便重复实验）────────────────────────────
    run_dir = f"./LM_llama3_8B/{args.task1}"
    os.makedirs(run_dir, exist_ok=True)

    LORA_ADAPTER_PATH = os.path.join(run_dir, "final_adapter")

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
    all_predict_results = []
    all_pred_tokens_corpus = []

    test_context = test_context[:100] # only eval on 100 samples for quick testing; set to None or remove this line for full evaluation
    # Setup Dataloader
    tokenizer.padding_side = 'left' 
    test_dataset = EmpathyDataset4Pred(
        contexts=test_context,
        targets=test_target,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_length=512,
    )
    collate_fn = PredictionDataCollator(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    processed_samples = 0
    for batch_idx_num, batch in enumerate(test_dataloader):
        generated_list, loss_list, ppl_list, sample_ppl_list, batch_predict_dicts = multi_turn_chat_with_ppl_batched(
            model=model,
            tokenizer=tokenizer,
            DEVICE=DEVICE,
            batch=batch,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        for b_i in range(len(generated_list)):
            generated = generated_list[b_i]
            reference = batch["reference"][b_i]
            history = batch["history"][b_i]
            ppl = ppl_list[b_i]
            sample_ppl = sample_ppl_list[b_i]
            loss = loss_list[b_i]
            sample_idx = batch["sample_idx"][b_i]

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
            all_predict_results.extend([batch_predict_dicts[b_i]])

            all_results.append({
                "id": sample_idx,
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
                    "my_bleu1": qshb1,
                    "my_bleu2": qshb2,
                }
            })
            
            processed_samples += 1
            if processed_samples % 100 == 0:
                print(f"Evaluated {processed_samples}/{len(test_context)} samples ...")
                print(f"[Round {processed_samples}] Average PPL: {ppl_total / processed_samples:.4f}")
                print(f"[Round {processed_samples}] Average Sample PPL: {sample_ppl_total / processed_samples:.4f}")
                print(f"[Round {processed_samples}] Average BLEU-1: {bleu1_total / processed_samples:.4f}")
                print(f"[Round {processed_samples}] Average BLEU-2: {bleu2_total / processed_samples:.4f}")

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
    
    output_jsonl_path = os.path.join(run_dir, f"eval_new_results_{lr}_{int(ratio*100)}_eval_{args.task1}_{args.task2}.jsonl")
    output_predict_json_path = os.path.join(run_dir, f"eval_pred_top10_{lr}_{int(ratio*100)}_eval_{args.task1}_{args.task2}.json")
    
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
    with open(output_predict_json_path, "w", encoding="utf-8") as f:
        json.dump(all_predict_results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_jsonl_path}")
    
