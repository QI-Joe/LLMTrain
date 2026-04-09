import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
from ZGeneration.train_gen_fast_LM import calculate_per_sample_ppl
from LM_Code.data_module import EmpathyDatasetForPrediction as EmpathyDataset4Pred
from LM_Code.data_module import PredictionDataCollator
from LM_Code.My_qwenLM import compute_bleu, compute_sentence_bleu
from src_analysis.metrics_func import calc_distinct
from bert_score import score as bert_score


def multi_turn_chat_with_ppl_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch: dict,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    model.eval()

    context_input_ids = batch["context_input_ids"].to(device)
    context_attention_mask = batch["context_attention_mask"].to(device)
    full_input_ids = batch["full_input_ids"].to(device)
    full_attention_mask = batch["full_attention_mask"].to(device)
    full_labels = batch["full_labels"].to(device)

    with torch.no_grad():
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
        )

    prompt_len = context_input_ids.shape[-1]
    new_tokens_tensor = output_ids.sequences[:, prompt_len:]
    generated_responses = tokenizer.batch_decode(new_tokens_tensor, skip_special_tokens=True)

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
    losses = losses.view(shift_logits.size(0), -1)

    mask = (shift_labels != -100).float()
    seq_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    ppl_list = torch.exp(seq_losses).cpu().numpy().tolist()
    loss_list = seq_losses.cpu().numpy().tolist()

    return generated_responses, loss_list, ppl_list, sample_ppl_list


def compute_bertscore_batch(
    candidates: list[str],
    references: list[str],
    model_name: str,
    batch_size: int,
    device: str,
    use_idf: bool,
    rescale_with_baseline: bool,
):

    all_p, all_r, all_f1 = [], [], []
    total = len(candidates)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        p, r, f1 = bert_score(
            candidates[start:end],
            references[start:end],
            model_type=model_name,
            lang="en",
            device=device,
            batch_size=batch_size,
            idf=use_idf,
            rescale_with_baseline=rescale_with_baseline,
            use_fast_tokenizer=False,
            verbose=False,
        )
        all_p.extend(p.cpu().tolist())
        all_r.extend(r.cpu().tolist())
        all_f1.extend(f1.cpu().tolist())

    return all_p, all_r, all_f1


def compute_bartscore_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sources: list[str],
    targets: list[str],
    device: str,
    batch_size: int,
    max_length: int,
):
    model.eval()
    all_scores = []

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    for start in range(0, len(sources), batch_size):
        end = min(start + batch_size, len(sources))
        src_batch = sources[start:end]
        tgt_batch = targets[start:end]

        inputs = tokenizer(
            src_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        if hasattr(tokenizer, "as_target_tokenizer"):
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    tgt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
        else:
            labels = tokenizer(
                tgt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        labels_ids = labels["input_ids"].to(device)
        labels_ids[labels_ids == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_ids[..., 1:].contiguous()
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(shift_logits.size(0), -1)
        mask = (shift_labels != -100).float()
        seq_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)

        batch_scores = (-seq_losses).cpu().numpy().tolist()
        all_scores.extend(batch_scores)

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation with BERTScore and BARTScore.")
    parser.add_argument("--task1", type=str, default="evaluation", help="Run folder name under LM_llama3_8B.")
    parser.add_argument("--task2", type=str, default="kv_cache_have", help="Extra tag for logs.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Dataset ratio (unused for eval-only).")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--eval_limit", type=int, default=-1, help="Limit test samples; -1 for full set.")
    parser.add_argument("--base_model", type=str, default="../../LLModel/llama3.1-8B-Instruct")
    parser.add_argument("--bert_model", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--bart_model", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--bert_idf", action="store_true", help="Enable IDF weighting for BERTScore.")
    parser.add_argument("--bert_rescale", action="store_true", help="Enable baseline rescaling for BERTScore.")
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SYSTEM_PROMPT = (
        "You are the assistant trying to show your empathy to the user during the "
        "conversation. Please don't over reply to the user's message (i.e., no need to use so many sentences.). "
        "Reply to the user's message as naturally as possible."
    )

    root_data = "data/ED"
    test_context = np.load(os.path.join(root_data, "sys_dialog_texts.test.npy"), allow_pickle=True)
    test_target = np.load(os.path.join(root_data, "sys_target_texts.test.npy"), allow_pickle=True)
    test_sit = np.load(os.path.join(root_data, "sys_situation_texts.test.npy"), allow_pickle=True)
    test_emo = np.load(os.path.join(root_data, "sys_emotion_texts.test.npy"), allow_pickle=True)

    if args.eval_limit is not None and args.eval_limit > 0:
        test_context = test_context[: args.eval_limit]
        test_target = test_target[: args.eval_limit]
        test_sit = test_sit[: args.eval_limit]
        test_emo = test_emo[: args.eval_limit]

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        cache_dir="./llama3-8B/",
        force_download=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./llama3-8B/",
        force_download=False,
    )

    run_dir = f"./LM_llama3_8B/{args.task1}"
    os.makedirs(run_dir, exist_ok=True)
    lora_adapter_path = os.path.join(run_dir, "final_adapter")

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

    print(f"Loading LoRA adapter from {lora_adapter_path}")
    model.load_adapter(lora_adapter_path, adapter_name="default")
    model.set_adapter("default")

    tokenizer.padding_side = "left"
    test_dataset = EmpathyDataset4Pred(
        contexts=test_context,
        targets=test_target,
        situations=test_sit,
        emotion_labels=test_emo,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_length=512,
    )
    collate_fn = PredictionDataCollator(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    total_dataset = len(test_dataset)

    ppl_total = 0.0
    sample_ppl_total = 0.0
    bleu1_total = 0.0
    bleu2_total = 0.0
    bleu3_total = 0.0
    bleu4_total = 0.0
    my_bleu1_total = 0.0
    my_bleu2_total = 0.0

    all_results = []
    all_generated = []
    all_references = []

    processed_samples = 0
    for batch in test_dataloader:
        generated_list, loss_list, ppl_list, sample_ppl_list = multi_turn_chat_with_ppl_batched(
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
            batch=batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        for b_i in range(len(generated_list)):
            generated = generated_list[b_i]
            reference = batch["reference"][b_i]
            history = batch["history"][b_i]
            ppl = ppl_list[b_i]
            sample_ppl = sample_ppl_list[b_i]
            sample_idx = batch["sample_idx"][b_i]

            bleu1, bleu2, bleu3, bleu4 = compute_sentence_bleu(generated, reference)
            # my_b1, my_b2 = compute_bleu(generated, reference)

            bleu1_total += bleu1
            bleu2_total += bleu2
            bleu3_total += bleu3
            bleu4_total += bleu4
            ppl_total += ppl
            sample_ppl_total += sample_ppl

            all_generated.append(generated)
            all_references.append(reference)

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
                }
            })

            processed_samples += 1
            if processed_samples % 100 == 0:
                print(f"Evaluated {processed_samples}/{total_dataset} samples ...")

    corpus_dist_1, corpus_dist_2 = calc_distinct(all_generated, tokenizer)

    total_samples = max(len(all_results), 1)
    print(f"Average PPL: {ppl_total / total_samples:.4f}")
    print(f"Average Sample PPL: {sample_ppl_total / total_samples:.4f}")
    print(f"Average BLEU-1: {bleu1_total / total_samples:.4f}")
    print(f"Average BLEU-2: {bleu2_total / total_samples:.4f}")
    print(f"Average BLEU-3: {bleu3_total / total_samples:.4f}")
    print(f"Average BLEU-4: {bleu4_total / total_samples:.4f}")
    print(f"Corpus Dist-1: {corpus_dist_1:.4f}")
    print(f"Corpus Dist-2: {corpus_dist_2:.4f}")

    print("Loading BERTScore model...")
    bert_p, bert_r, bert_f1 = compute_bertscore_batch(
        all_generated,
        all_references,
        model_name=args.bert_model,
        batch_size=args.score_batch_size,
        device=DEVICE,
        use_idf=args.bert_idf,
        rescale_with_baseline=args.bert_rescale,
    )

    print("Loading BARTScore model...")
    bart_tokenizer = AutoTokenizer.from_pretrained(args.bart_model)
    if bart_tokenizer.pad_token is None:
        bart_tokenizer.pad_token = bart_tokenizer.eos_token
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(args.bart_model).to(DEVICE)

    bart_gen_given_ref = compute_bartscore_batch(
        bart_model,
        bart_tokenizer,
        sources=all_references,
        targets=all_generated,
        device=DEVICE,
        batch_size=args.score_batch_size,
        max_length=512,
    )
    bart_ref_given_gen = compute_bartscore_batch(
        bart_model,
        bart_tokenizer,
        sources=all_generated,
        targets=all_references,
        device=DEVICE,
        batch_size=args.score_batch_size,
        max_length=512,
    )

    for idx, item in enumerate(all_results):
        item["metrics"]["dist1_corpus"] = corpus_dist_1
        item["metrics"]["dist2_corpus"] = corpus_dist_2
        item["metrics"]["bertscore_p"] = bert_p[idx]
        item["metrics"]["bertscore_r"] = bert_r[idx]
        item["metrics"]["bertscore_f1"] = bert_f1[idx]
        item["metrics"]["bartscore_gen_given_ref"] = bart_gen_given_ref[idx]
        item["metrics"]["bartscore_ref_given_gen"] = bart_ref_given_gen[idx]

    print(f"Average BERTScore P: {sum(bert_p) / len(bert_p):.4f}")
    print(f"Average BERTScore R: {sum(bert_r) / len(bert_r):.4f}")
    print(f"Average BERTScore F1: {sum(bert_f1) / len(bert_f1):.4f}")
    print(f"Average BARTScore gen|ref: {sum(bart_gen_given_ref) / len(bart_gen_given_ref):.4f}")
    print(f"Average BARTScore ref|gen: {sum(bart_ref_given_gen) / len(bart_ref_given_gen):.4f}")

    output_jsonl_path = os.path.join(run_dir, "bart_bert_eval.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Evaluation results saved to {output_jsonl_path}")
