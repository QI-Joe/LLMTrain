import os
import sys
import json
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import argparse

from data_module import EmpathyDataset, IAMMDataCollator, EMOTION_MAP
from train_module import EmotionHead, IAMMTrainer, multi_turn_chat_with_ppl, compute_sentence_bleu, compute_bleu

def run_training(model, tokenizer, train_data, args_dict):
    print("Building training dataset ...")
    
    contexts, targets, situations, emotions = train_data
    
    train_dataset = EmpathyDataset(
        contexts=contexts,
        targets=targets,
        situations=situations,
        emotion_labels=emotions,
        tokenizer=tokenizer,
        system_prompt=args_dict['SYSTEM_PROMPT'],
        max_length=512,
        sit_max_length=128
    )
    
    print(f"Training samples: {len(train_dataset)}\n")

    training_args = TrainingArguments(
        output_dir=os.path.join(args_dict['BASE_PATH'], args_dict['FOLDER_NAME']),
        seed=args_dict['SEED'],
        num_train_epochs=3,
        per_device_train_batch_size=2, # 2 is original setting
        gradient_accumulation_steps=8, # 8 is original setting, effective batch size = 16
        learning_rate=args_dict['lr'],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    emo_head = EmotionHead(hidden_size=model.config.hidden_size, num_emotions=len(EMOTION_MAP))
    data_collator = IAMMDataCollator(tokenizer=tokenizer)

    trainer = IAMMTrainer(
        emo_head=emo_head,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting IAMM LoRA fine-tuning ...")
    trainer.train()
    print("Fine-tuning complete.\n")

    model.save_pretrained(args_dict['LORA_ADAPTER_PATH'])
    tokenizer.save_pretrained(args_dict['LORA_ADAPTER_PATH'])
    emo_head.save(args_dict['LORA_ADAPTER_PATH'])
    print(f"LoRA adapter saved → {args_dict['LORA_ADAPTER_PATH']}\n")
    
    return emo_head

if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Train and evaluate the IAMM model.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Ratio of training data to use.")
    parser.add_argument("--new_model_train", action="store_true", help="Whether to train a new model or load existing adapter.")
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
    if not os.path.exists(LORA_ADAPTER_PATH) or args.new_model_train:
        emo_head = run_training(model, tokenizer, train_data, args_dict)
    else:
        print(f"Found existing LoRA adapter at '{LORA_ADAPTER_PATH}', skipping training.\n")
        model.load_adapter(LORA_ADAPTER_PATH, adapter_name="default")
        model.set_adapter("default")
        
        # We need the emo head even if skip training
        emo_head = EmotionHead(hidden_size=model.config.hidden_size, num_emotions=len(EMOTION_MAP))
        emo_head.load_state_dict(torch.load(os.path.join(LORA_ADAPTER_PATH, 'emotion_head.pt'), map_location=DEVICE))

    # --- Evaluation ---
    ppl_total, sample_ppl_total = 0.0, 0.0
    bleu1_total, bleu2_total, bleu3_total, bleu4_total = 0.0, 0.0, 0.0, 0.0
    qshbleu1, qshbleu2 = 0.0, 0.0
    emo_correct = 0

    all_results = []

    print("Evaluating...")
    for i in range(len(test_context)):
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for j in range(len(test_context[i])):
            role = "user" if j % 2 == 0 else "assistant"
            history.append({"role": role, "content": test_context[i][j]})

        reference = test_target[i]
        situation = test_sit[i]
        emotion_label_str = test_emo[i]
        emotion_idx = EMOTION_MAP.get(emotion_label_str, 0)

        generated, loss, ppl, sample_ppl, emo_pred = multi_turn_chat_with_ppl(
            model=model,
            tokenizer=tokenizer,
            DEVICE=DEVICE,
            history=history,
            reference_answer=reference,
            situation_text=situation,
            emo_head=emo_head.to(DEVICE),
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        if emo_pred == emotion_idx:
            emo_correct += 1

        bleu1, bleu2, bleu3, bleu4 = compute_sentence_bleu(generated, reference)
        qshb1, qshb2 = compute_bleu(generated, reference)

        bleu1_total += bleu1; bleu2_total += bleu2; bleu3_total += bleu3; bleu4_total += bleu4
        qshbleu1 += qshb1; qshbleu2 += qshb2
        ppl_total += ppl; sample_ppl_total += sample_ppl

        all_results.append({
            "id": i,
            "reference": reference,
            "generated": generated,
            "emotion_label": emotion_idx,
            "emotion_pred": emo_pred,
            "metrics": {
                "ppl": ppl, "sample_ppl": sample_ppl,
                "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
                "my_bleu1": qshb1, "my_bleu2": qshb2,
            }
        })

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
    print(f"Average PPL: {ppl_total / len(test_context):.4f}")
    print(f"Average Sample PPL: {sample_ppl_total / len(test_context):.4f}")
    print(f"Average BLEU-1: {bleu1_total / len(test_context):.4f}")
    print(f"Average BLEU-2: {bleu2_total / len(test_context):.4f}")
    print(f"Average My BLEU-1: {qshbleu1 / len(test_context):.4f}")
    print(f"Average My BLEU-2: {qshbleu2 / len(test_context):.4f}")
    print(f"Corpus Dist-1: {corpus_dist_1:.4f}")
    print(f"Corpus Dist-2: {corpus_dist_2:.4f}")

    for item in all_results:
        item["metrics"]["dist1_corpus"] = corpus_dist_1
        item["metrics"]["dist2_corpus"] = corpus_dist_2
        item["metrics"]["emotion_accuracy"] = emo_correct / len(test_context)
    
    output_jsonl_path = os.path.join(BASE_PATH, FOLDER_NAME, f"eval_results_{FOLDER_NAME}.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Evaluation results saved to {output_jsonl_path}")
