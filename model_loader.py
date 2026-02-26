import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# from src.utils.data.loader import Dataset, prepare_data_seq, load_dataset
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
import os
from typing import List, Dict, Tuple, Optional, Any
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


class LlamaModelDownload(nn.Module):
    def __init__(self, model_path: str, device, quant: bool = True, download_str: str = r'~/Documents/LLModel'):
        super().__init__()
        self.model_path = model_path
        # Expand ~ to actual home directory path
        self.dl_path = os.path.expanduser(download_str)
        self.device = device
        self.is_local = False
        self.quant = quant

    def load(self, view_hidden_size: bool = False):
        model_path = self.model_path
        if '/' not in self.model_path or len(self.model_path.split("/")) <= 1: # huggingface model in shape of param1/param2
            self.is_local = True
            model_path = os.path.join(self.dl_path, self.model_path)
            print("model in local")
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,  # Llama3建议使用slow tokenizer
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        
        # 配置 4-bit 量化（减少显存使用）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # 启用 4-bit 加载
            bnb_4bit_quant_type="nf4",      # 使用 NF4 数据类型（精度更高）
            bnb_4bit_compute_dtype=torch.float16, # 计算时使用 float16，防止溢出
            bnb_4bit_use_double_quant=True, # 二次量化，进一步节省显存
        )

        # 配置设备分配（关键：确保模型加载到正确的GPU）
        if "cuda" in str(self.device):
            # 提取GPU ID
            if ":" in str(self.device):
                gpu_id = int(str(self.device).split(":")[1])
            else:
                gpu_id = 0
            
            # 使用max_memory指定具体GPU
            max_memory = {gpu_id: "22GB", "cpu": "30GB"}
            device_map_config = "auto"
            print(max_memory)
        else:
            max_memory = None
            device_map_config = "cpu"
        # 加载模型（应用量化配置）
        model = AutoModel.from_pretrained(
            model_path,
            quantization_config=bnb_config if self.quant else None,  # 🔑 关键：传入量化配置
            device_map=device_map_config,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        # model.to(self.device)
        if view_hidden_size:
            config = AutoConfig.from_pretrained(model_path)
            print(f"The Attention path of model {model_path} is: {config.hidden_size}")
        
        return tokenizer, model
    
    def store(self, model: AutoModel, token: AutoTokenizer):
        my_special_tokens = {
            "pad_token": "PAD",
            "eos_token": "EOS", 
            "unk_token": "UNK",
            # 对于 SOS, USR, SYS, CLS，我们将它们放入 additional_special_tokens
            "additional_special_tokens": ["SOS", "USR", "SYS", "CLS"]
        }
        num_add_toks = token.add_special_tokens(my_special_tokens)
        len_token = len(token)
        print(f"Added {num_add_toks} tokens")
        print(f"New vocab size: {len_token}")
        
        print(f"PAD token id: {token.pad_token_id}")
        print(f"USR token id: {token.convert_tokens_to_ids('USR')}")
        model.resize_token_embeddings(len_token)
        
        model_path = os.path.join(self.dl_path, self.model_path.split('/')[-1])
        os.makedirs(model_path, exist_ok=True)
        
        model.save_pretrained(model_path)
        token.save_pretrained(model_path)
        
        return model_path
    
    def start(self, view_hidden_size: bool = False) -> Tuple[AutoModel, AutoTokenizer]:
        tokenizer, model = self.load(view_hidden_size)
        if self.is_local:
            return model, tokenizer
        modelpath = self.store(model, tokenizer)
        print(f'New model download at {modelpath}')
        return model, tokenizer


class LlamaClassification(nn.Module):
    def __init__(self, hidden_size, model: AutoModel, tokenizer: AutoTokenizer, num_emos: int = 32):
        super(LlamaClassification, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        self.model = self._apply_lora(self.model)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_emos)
        self.hidden_size, self.num_emo = hidden_size, num_emos
        
        self.loss_fct = nn.CrossEntropyLoss()
        
        # 关键：将分类头移到模型所在的设备
        # 量化模型通过 device_map 已经在 GPU 上，需要让分类头也在同一设备
        model_device = next(self.model.parameters()).device
        self.classifier = self.classifier.to(model_device)
        self.dropout = self.dropout.to(model_device)
    
    def _apply_lora(self, model):
        """应用 LoRA 配置"""
        # 准备模型用于 k-bit 训练
        model = prepare_model_for_kbit_training(model)
        
        # 配置 LoRA
        lora_config = LoraConfig(
            r=16,                      # LoRA rank，越大精度越高但显存越多
            lora_alpha=32,             # LoRA 缩放参数
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama3 attention 层
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"  # 分类任务用这个
        )
        
        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数量
        return model
        
    def forward(self, data: Dict[str, Optional[torch.Tensor | str | torch.LongTensor]]):
        '''
        Be aware, emotion == label
        
        :param data: 同IAMM数据类型相似
        :type data: Dict[str, Optional[torch.Tensor | str | torch.LongTensor]]
        '''
        
        context: torch.Tensor = data['input_ids']
        emotion = data['labels']  # 修复：改为labels与data_loader一致
        src_mask = data['attention_mask']
        
        output: torch.Tensor = self.model.forward(
            input_ids = context, attention_mask = src_mask
        )
        
        sequence_output = output.last_hidden_state
        last_token_indices = src_mask.sum(1) - 1
        batch_size = context.shape[0]
        
        # 获取最后一个有效token的表示（确保设备一致）
        sentence_reprsentation = sequence_output[torch.arange(batch_size, device=sequence_output.device), last_token_indices]
        
        x = self.dropout(sentence_reprsentation)
        logits: torch.Tensor = self.classifier(x)
        
        loss = self.loss_fct(logits.view(-1, self.num_emo), emotion.view(-1))
        return logits, loss
    

if __name__ == "__main__":
    model1 = 'Llama-3.3-8B-Instruct'
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"here is the model1 {model1}")
    modeLoader = LlamaModelDownload(model1, device)
    modeLoader.start(True)

