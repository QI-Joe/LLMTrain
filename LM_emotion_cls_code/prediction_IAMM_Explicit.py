import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import argparse

from LM_Code.data_module import EMOTION_MAP, EmpathyDatasetForPrediction, PredictionDataCollator
from LM_Code.train_module import EmotionHead
from torch.utils.data import DataLoader

REVERSE_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

def multi_turn_chat_with_ppl_batched(
    model,
    tokenizer,
    DEVICE,
    batch,
    emo_head,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    model.eval()
    emo_head.eval()
    
    context_input_ids = batch["context_input_ids"].to(DEVICE)
    context_attention_mask = batch["context_attention_mask"].to(DEVICE)
    full_input_ids = batch["full_input_ids"].to(DEVICE)
    full_attention_mask = batch["full_attention_mask"].to(DEVICE)
    full_labels = batch["full_labels"].to(DEVICE)
    
    situation_input_ids = batch["situation_input_ids"].to(DEVICE)
    situation_attention_mask = batch["situation_attention_mask"].to(DEVICE)
    emotion_labels = batch["emotion_label"].to(DEVICE)

    with torch.no_grad():
        # Set manual seeds inside just like Explict_Prediction_topk.py
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        output_ids = model.generate(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = context_input_ids.shape[-1]
    new_tokens_tensor = output_ids[:, prompt_len:]
    generated_responses = tokenizer.batch_decode(new_tokens_tensor, skip_special_tokens=True)
    
    # EMOTION PREDICTION
    with torch.no_grad():
        sit_outputs = model(
            input_ids=situation_input_ids,
            attention_mask=situation_attention_mask,
            output_hidden_states=True
        )
        last_hidden_states = sit_outputs.hidden_states[-1]
        emo_logits = emo_head(last_hidden_states, attention_mask=situation_attention_mask)
        top5_value, top_5_indices = torch.topk(emo_logits, k=5, dim=-1)
        batch_first5 = top_5_indices.cpu().numpy().tolist()
        emo_preds = torch.argmax(emo_logits, dim=-1).cpu().numpy().tolist()

    # Step 2: Compute PPL
    # with torch.no_grad():
    #     logits = model(
    #         input_ids=full_input_ids, 
    #         attention_mask=full_attention_mask
    #     ).logits

    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = full_labels[..., 1:].contiguous()

    # sample_ppl_list = calculate_per_sample_ppl(logits, full_labels)
    
    # loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    # losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # batch_size = full_input_ids.size(0)
    # losses = losses.view(batch_size, -1)
    
    # mask = (shift_labels != -100).float()
    # seq_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    # ppl_list = torch.exp(seq_losses).cpu().numpy().tolist()
    # loss_list = seq_losses.cpu().numpy().tolist()

    return generated_responses, emo_preds, batch_first5 #, loss_list, ppl_list, sample_ppl_list


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Train and evaluate the IAMM model.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Ratio of training data to use.")
    parser.add_argument("--task1", type=str, default="Gen", help="Task 1 name for folder naming.")
    parser.add_argument("--task2", type=str, default="ED", help="Task 2 name for folder naming.")
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
    SYSTEM_PROMPT = "You are the assistant trying to show your empathy to the user during the "\
                    "conversation. Please don't over reply to the user's message (i.e., no need to use so many sentences.). "\
                    "Reply to the user's message as naturally as possible."

    BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'llama3-8B')
    FOLDER_NAME = f"llama3_iamm_{args.task1}_{args.task2}_{args.lr}_{int(args.ratio*100)}"
    LORA_ADAPTER_PATH = os.path.join(BASE_PATH, FOLDER_NAME, "adapter")
    args_dict = {
        'lr': args.lr,
        'SYSTEM_PROMPT': SYSTEM_PROMPT,
        'SEED': SEED,
        'LORA_ADAPTER_PATH': LORA_ADAPTER_PATH,
        'BASE_PATH': BASE_PATH,
        'FOLDER_NAME': FOLDER_NAME
    }

    # --- Data Loading ---
    root_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ED'))
    train_context = np.load(os.path.join(root_data, 'sys_dialog_texts.train.npy'), allow_pickle=True)
    train_target  = np.load(os.path.join(root_data, 'sys_target_texts.train.npy'), allow_pickle=True)
    train_sit     = np.load(os.path.join(root_data, 'sys_situation_texts.train.npy'), allow_pickle=True)
    train_emo     = np.load(os.path.join(root_data, 'sys_emotion_texts.train.npy'), allow_pickle=True)

    test_context = np.load(os.path.join(root_data, 'sys_dialog_texts.test.npy'), allow_pickle=True)
    test_target  = np.load(os.path.join(root_data, 'sys_target_texts.test.npy'), allow_pickle=True)
    test_sit     = np.load(os.path.join(root_data, 'sys_situation_texts.test.npy'), allow_pickle=True)
    test_emo     = np.load(os.path.join(root_data, 'sys_emotion_texts.test.npy'), allow_pickle=True)
    
    # Data sub-sampling
    indices = np.random.choice(len(train_context), size=int(len(train_context) * args.ratio), replace=False)
    train_data = (
        train_context[indices],
        train_target[indices],
        train_sit[indices],
        train_emo[indices]
    )
    print(f"Loaded and sub-sampled training data: {indices.shape[0]} samples\n")

    # --- Model Loading ---
    print("Loading model ...")
    model_name = "../../LLModel/llama3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./llama3-8B/', force_download=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", cache_dir='./llama3-8B/', force_download=False
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print()

    # --- Training ---
    print(f"Found existing LoRA adapter at '{LORA_ADAPTER_PATH}', skipping training.\n")
    model.load_adapter(LORA_ADAPTER_PATH, adapter_name="default")
    model.set_adapter("default")
    
    # We need the emo head even if skip training
    emo_head = EmotionHead(hidden_size=model.config.hidden_size, num_emotions=len(EMOTION_MAP)).half()
    emo_head.load_state_dict(torch.load(os.path.join(LORA_ADAPTER_PATH, 'emotion_head.pt'), map_location=DEVICE))
    emo_head = emo_head.to(model.dtype)

    # --- Evaluation ---
    ppl_total, sample_ppl_total = 0.0, 0.0
    bleu1_total, bleu2_total, bleu3_total, bleu4_total = 0.0, 0.0, 0.0, 0.0
    qshbleu1, qshbleu2 = 0.0, 0.0
    emo_correct = 0

    all_results = []

    print("Evaluating...")
    tokenizer.padding_side = 'left' 
    test_prediction_dataset = EmpathyDatasetForPrediction(
        contexts=test_context,
        targets=test_target,
        situations=test_sit,
        emotion_labels=test_emo,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_length=512,
    )
    collate_fn = PredictionDataCollator(tokenizer)
    test_dataloader = DataLoader(test_prediction_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print("[Runing] Data preparation finished, lets go on batch evaluation ...")
    processed_samples = 0
    for batch_idx_num, batch in enumerate(test_dataloader):
        generated_list, emo_preds, batch_frist5_emo = multi_turn_chat_with_ppl_batched(
            model=model,
            tokenizer=tokenizer,
            DEVICE=DEVICE,
            batch=batch,
            emo_head=emo_head.to(DEVICE),
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        for b_i in range(len(generated_list)):
            generated = generated_list[b_i]
            reference = batch["reference"][b_i]
            history = batch["history"][b_i]
            sample_idx = batch["sample_idx"][b_i]
            
            emotion_idx = batch["emotion_label"][b_i].item()
            emo_pred = emo_preds[b_i]
            emo_top5 = batch_frist5_emo[b_i]

            if emo_pred == emotion_idx:
                emo_correct += 1

            all_results.append({
                "id": sample_idx,
                "history": history,
                "reference": reference,
                "generated": generated,
                "emotion_label": REVERSE_EMOTION_MAP.get(emotion_idx, str(emotion_idx)),
                "emotion_pred": REVERSE_EMOTION_MAP.get(emo_pred, str(emo_pred)),
                "emotion_top5_list": [REVERSE_EMOTION_MAP.get(idx, str(idx)) for idx in emo_top5],
            })
            
            processed_samples += 1
            if processed_samples % 50 == 0:
                print(f"Evaluated {processed_samples}/{len(test_context)} samples ...")

    unigrams = set()
    bigrams = set()
    total_unigrams = 0
    total_bigrams = 0
    
    for item in all_results:
        tokens = item["generated"].strip().split()
        for idx in range(len(tokens)):
            unigrams.add(tokens[idx])
            total_unigrams += 1
            if idx < len(tokens) - 1:
                bigrams.add((tokens[idx], tokens[idx+1]))
                total_bigrams += 1

    corpus_dist_1 = len(unigrams) / total_unigrams if total_unigrams > 0 else 0.0
    corpus_dist_2 = len(bigrams) / total_bigrams if total_bigrams > 0 else 0.0

    print(f"Emotion Accuracy: {emo_correct / len(test_context):.4f}")

    output_json_path = os.path.join(BASE_PATH, FOLDER_NAME, f"eval_EMO_LABEL_{FOLDER_NAME}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_json_path}")

