import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

EMOTION_MAP = {
    "afraid": 0, "angry": 1, "annoyed": 2, "anticipating": 3, "anxious": 4,
    "apprehensive": 5, "ashamed": 6, "caring": 7, "confident": 8, "content": 9,
    "devastated": 10, "disappointed": 11, "disgusted": 12, "embarrassed": 13,
    "excited": 14, "faithful": 15, "furious": 16, "grateful": 17, "guilty": 18,
    "hopeful": 19, "impressed": 20, "jealous": 21, "joyful": 22, "lonely": 23,
    "nostalgic": 24, "prepared": 25, "proud": 26, "sad": 27, "sentimental": 28,
    "surprised": 29, "terrified": 30, "trusting": 31
}

class EmpathyDataset(Dataset):
    """
    Formats (context, target) pairs into full dialogue sequences.
    Labels for context are set to -100 to compute loss only on target.
    Also includes situations and emotion labels for IAMM.
    """
    def __init__(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        situations: np.ndarray,
        emotion_labels: np.ndarray,
        tokenizer,
        system_prompt: str,
        max_length: int = 512,
        sit_max_length: int = 128
    ):
        self.samples = []
        skipped = 0

        for context, target, situation, emotion in zip(contexts, targets, situations, emotion_labels):
            history = [{"role": "system", "content": system_prompt}]
            for j, utt in enumerate(context):
                role = "user" if j % 2 == 0 else "assistant"
                history.append({"role": role, "content": utt})

            context_text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                history + [{"role": "assistant", "content": target}],
                tokenize=False,
                add_generation_prompt=False,
            )

            full_ids = tokenizer(full_text)["input_ids"]
            context_len = len(tokenizer(context_text)["input_ids"])

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            if context_len >= len(full_ids):
                skipped += 1
                continue

            labels = [-100] * context_len + full_ids[context_len:]
            full_mask = [1] * len(full_ids)

            sit_inputs = tokenizer(
                situation,
                max_length=sit_max_length,
                truncation=True
            )
            
            emotion_idx = EMOTION_MAP.get(emotion, 0) # Fallback to 0 if not found

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                # "attention_mask": torch.tensor(full_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "situation_input_ids": torch.tensor(sit_inputs["input_ids"], dtype=torch.long),
                "situation_attention_mask": torch.tensor(sit_inputs["attention_mask"], dtype=torch.long),
                "emotion_label": emotion_idx
            })

        if skipped:
            print(f"[Dataset] Skipped {skipped} samples (context >= max_length).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class IAMMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8,label_pad_token_id=-100)
        
    def __call__(self, features):
        main_features = [
            {"input_ids": f["input_ids"], "labels": f["labels"]}
            for f in features
        ]
        
        # 调用 base_collator 后，batch 字典中会自动多出一个补齐后的 "attention_mask" 键 
        batch = self.base_collator(main_features)
        
        sit_features = [
            {"input_ids": f["situation_input_ids"], "attention_mask": f["situation_attention_mask"]}
            for f in features
        ]
        sit_batch = self.tokenizer.pad(
            sit_features,
            padding=True,
            return_tensors="pt"
        )
        
        batch["situation_input_ids"] = sit_batch["input_ids"]
        batch["situation_attention_mask"] = sit_batch["attention_mask"]
        batch["emotion_label"] = torch.tensor([f["emotion_label"] for f in features], dtype=torch.long)
        
        return batch
