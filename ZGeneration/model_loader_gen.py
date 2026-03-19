import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import os
from typing import Tuple, Any
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class GenModelLoader(nn.Module):
    def __init__(self, model_path: str, device, accerlator, download_str: str = r'~/Documents/LLModel'):
        super().__init__()
        self.model_path = model_path
        self.dl_path = os.path.expanduser(download_str)
        self.device = device
        self.is_local = False
        self.accerlator: Any = accerlator
        
        # Enforce quantization (User Requirement: quant setting has to be activated)
        self.quant = True 

    def load_tokenizer_only(self):
        model_path = self.model_path
        if '/' not in self.model_path or len(self.model_path.split("/")) <= 1: 
            self.is_local = True
            model_path = os.path.join(self.dl_path, self.model_path)
            print("Model found locally at:", model_path)
        return AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True, 
            trust_remote_code=True
        )

    def load(self, raw_model):
        model_path = self.model_path
        if '/' not in self.model_path or len(self.model_path.split("/")) <= 1: 
            self.is_local = True
            model_path = os.path.join(self.dl_path, self.model_path)
            print("Model found locally at:", model_path)
            
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True, 
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            
        tokenizer.padding_side = 'right' # Better for generation usually, but training implies right padding often.
        # tokenizer.padding_side = 'right'

        # BitsAndBytes Config (Enforced)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Device Map Config
        # Note: If CUDA_VISIBLE_DEVICES is set (as per Accelerator plan), 'cuda:0' is the target.
        # With BitsAndBytes quantization, device_map="auto" is REQUIRED for proper memory management
        # Using a specific device string like "cuda:0" causes inefficient placement and slow inference
        if "cuda" in str(self.device):
            device_map_config = "auto"  # CRITICAL: Use "auto" with quantization for proper performance
        else:
            device_map_config = "cpu"

        print(f"Loading Model with Quantization: {self.quant}, device_map: {device_map_config}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map_config,
            trust_remote_code=True,
            torch_dtype=torch.float16, 
        )
        
        # CRITICAL: Always resize embeddings if pad token was added (vocab size changed)
        # This must happen BEFORE LoRA to ensure base model embeddings match tokenizer
        if tokenizer.pad_token == '<|pad|>':
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized model embeddings: {model.get_input_embeddings().weight.shape[0]} to match tokenizer vocab size: {len(tokenizer)}")
        
        # Apply LoRA (only for fine-tuning, not raw model)
        if not raw_model:
            model = self._apply_lora(model)
        
        return tokenizer, model

    def _apply_lora(self, model):
        """Apply LoRA Configuration for Generation Task"""
        model = prepare_model_for_kbit_training(model)
        
        # LoRA for Causal LM
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Extended for better Gen
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" 
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def start(self, raw_model=False) -> Tuple[AutoTokenizer, object]:
        tokenizer, model = self.load(raw_model=raw_model)
        # Ensure special tokens are managed (store logic simplified here)
        return model, tokenizer
