import math
import torch
import torch.nn as nn
from collections import Counter
from transformers import Trainer
import nltk
from nltk.translate.bleu_score import sentence_bleu
import torch.nn.functional as F

class EmotionHead(nn.Module):
    def __init__(self, hidden_size, num_emotions):
        super().__init__()
        self.attention_layer2 = nn.Linear(hidden_size, hidden_size)
        self.attention_v2 = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer2 = nn.Linear(hidden_size, num_emotions)
    
    def forward(self, enc_outputs, attention_mask=None):
        # PEFT typically casts hidden states to float32 for numerical stability. 
        # We cast it back to the same dtype as our linear layers.
        enc_outputs = enc_outputs.to(self.attention_layer2.weight.dtype)
        projected = self.attention_layer2(enc_outputs)
        projected = nn.Tanh()(projected)
        attn_logits = self.attention_v2(projected).squeeze(2) # (B, L)
        
        if attention_mask is not None:
             mask_val = (1.0 - attention_mask.float()) * -10000.0
             attn_logits = attn_logits + mask_val.to(attn_logits.dtype)
        
        scores = F.softmax(attn_logits, dim=-1).unsqueeze(1) # (batch_size, 1, seq_len)
        
        hidden_x = torch.bmm(scores, enc_outputs).squeeze(1)
        
        x = nn.Tanh()(self.hidden_layer2(hidden_x))
        emo_logits = self.output_layer2(x)
        return emo_logits
    
    def save(self, save_directory):
        torch.save(self.state_dict(), f"{save_directory}/emotion_head.pt")

class IAMMTrainer(Trainer):
    def __init__(self, emo_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emo_head = emo_head.to(self.args.device)
        self.loss_fct = nn.CrossEntropyLoss()
        
        # Lists to store epoch predictions
        self.epoch_emo_preds = []
        self.epoch_emo_labels = []
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        
        situation_ids = inputs.pop("situation_input_ids")
        situation_mask = inputs.pop("situation_attention_mask")
        emotion_labels = inputs.pop("emotion_label")
        
        # Forward pass 1: Context (Text Generation)
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            use_cache=False,
        )
        ctx_loss = outputs.loss
        
        # Forward pass 2: Situation (Emotion Prediction)
        sit_outputs = model(
            input_ids=situation_ids,
            attention_mask=situation_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden_states = sit_outputs.hidden_states[-1]
        
        emo_logits = self.emo_head(last_hidden_states, attention_mask=situation_mask)
        sit_emo_loss = self.loss_fct(emo_logits, emotion_labels)
        
        # We append our custom data to the outputs object to have access to them later via HF Trainer's flow.
        outputs.emo_logits = emo_logits
        outputs.emotion_labels = emotion_labels
        
        total_loss = ctx_loss + sit_emo_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

    def create_optimizer(self):
        if self.optimizer is None:
            params = list(self.model.parameters()) + list(self.emo_head.parameters())
            from torch.optim import AdamW
            self.optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        return self.optimizer


# ─── PPL Calculation Alternative ──────────────────────────────────────────────
def calculate_per_sample_ppl(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    
    valid_lengths = (shift_labels != -100).sum(dim=1)
    sample_losses = loss.sum(dim=1) / valid_lengths.float()
    return torch.exp(sample_losses).tolist()

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
        emo_preds = torch.argmax(emo_logits, dim=-1).cpu().numpy().tolist()

    # Step 2: Compute PPL
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
    
    batch_size = full_input_ids.size(0)
    losses = losses.view(batch_size, -1)
    
    mask = (shift_labels != -100).float()
    seq_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    ppl_list = torch.exp(seq_losses).cpu().numpy().tolist()
    loss_list = seq_losses.cpu().numpy().tolist()

    return generated_responses, loss_list, ppl_list, sample_ppl_list, emo_preds


def _modified_precision(hypothesis, reference, n):
    hyp_ngrams = (
        Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        if len(hypothesis) >= n else Counter()
    )
    ref_ngrams = (
        Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        if len(reference) >= n else Counter()
    )

    numerator = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
    denominator = max(1, sum(hyp_ngrams.values()))
    if numerator == 0:
        numerator = 0.1
    return numerator / denominator

def compute_sentence_bleu(hypothesis_str, reference_str):
    hyp = hypothesis_str.strip().split()
    ref = reference_str.strip().split()
    if not hyp:
        return 0.0, 0.0, 0.0, 0.0

    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
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
    
    if len(ref_toks) == 0 or len(pred_toks) == 0:
        b1, b2 = 0.0, 0.0
    else:
        b1 = sentence_bleu([ref_toks], pred_toks, weights=(1, 0, 0, 0))
        b2 = sentence_bleu([ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0))
    
    return b1, b2
