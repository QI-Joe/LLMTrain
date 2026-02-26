import os
from typing import List, Dict, Optional, Tuple
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from model_loader import LlamaModelDownload, LlamaClassification
from data_loader_llama3 import get_dataloader
from config_llama3 import TrainingConfig
from data_loader_llama3 import EMOTION_MAP

# 32个情感标签
EMOTION_LIST = list(EMOTION_MAP.keys())


def load_checkpoint_for_inference(checkpoint_dir, device='cuda:3'):
    """从检查点加载模型用于推理"""
    
    parent_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
    config_path = os.path.join(parent_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Loading config: {config['model_name']}, hidden_size={config['hidden_size']}")
    
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True, trust_remote_code=True)
    
    model_downloader = LlamaModelDownload(config['model_name'], device=device, quant=True)
    base_model, tokenizer = model_downloader.start()
    tokenizer.padding_side = 'right'

    
    model = LlamaClassification(
        hidden_size=config['hidden_size'],
        model=base_model,
        tokenizer=tokenizer,
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model = model.to(device)
    
    print(f"✓ Model loaded at step {checkpoint['global_step']}")
    
    return model, tokenizer, config


def prepare_data_loaders(config_dict, tokenizer):
    """准备数据加载器"""
    
    config = TrainingConfig()
    config.data_path = config_dict['data_path']
    config.prompt_key = config_dict['prompt_key']
    config.dialogue_window = config_dict['dialogue_window']
    config.batch_size = config_dict['batch_size']
    config.max_seq_length = config_dict['max_seq_length']
    config.num_workers = config_dict['num_workers']
    config.few_shot = config_dict['few_shot']
    config.shots_per_class = config_dict['shots_per_class']
    config.semi_supervised = config_dict['semi_supervised']
    config.semi_ratio = config_dict['semi_ratio']
    config.fast_train = config_dict['fast_train']
    
    _, val_loader, test_loader = get_dataloader(tokenizer=tokenizer, config=config)
    
    return val_loader, test_loader

def get_core_intel(folder_dir: str, core_file_list: List[str] = ['summary.txt', 'training.log']) -> Dict[str, any]:
    """
    Extract key information from experiment output files using regex.
    Note:
        Read the regex tutorial in markdown/regex_tutorial.md to learn how to implement
        the regex patterns needed for this function.
    """
    summary_path = os.path.join(folder_dir, core_file_list[0])
    training_log_path = os.path.join(folder_dir, core_file_list[1])
    
    result = {}
    
    # ============================================================
    # Part 1: Parse summary.txt for test/validation results
    # ============================================================
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = r"(\w+):\s?(\d*\.\d*)"
        val_list = re.findall(pattern, content)
        test_val_split = len(val_list) // 2
        for idx, (key, value) in enumerate(val_list):
            result[key if idx >= test_val_split else 'test_'+key] = float(value)
        
        print(f"✓ Parsed summary.txt: {summary_path}")
    else:
        print(f"⚠️ summary.txt not found: {summary_path}")
    
    # ============================================================
    # Part 2: Parse training.log for dataset sizes
    # ============================================================
    if os.path.exists(training_log_path):
        with open(training_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        train_samples_pattern = r"Train samples:\s?(\d+)"  # Your regex here        
        val_samples_pattern = r"Val samples:\s?(\d+)"
        test_samples_pattern = r"Test samples:\s?(\d+)"  # Your regex here
        
        result["train_sample"] = int(re.search(train_samples_pattern, content).group(1))
        result["val_sample"] = int(re.search(val_samples_pattern, content).group(1))
        result["test_sample"] = int(re.search(test_samples_pattern, content).group(1))
        print(f"✓ Parsed training.log: {training_log_path}")
    else:
        print(f"⚠️ training.log not found: {training_log_path}")
    
    return result

def txt_format_record(output_file: str, device, core_intel: Dict[str, Optional[float | int]], config: Dict[str, Optional[str|int]], pred_target_text_list: List[Optional[List[int] | List[str]]]):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Total samples: {len(pred_target_text_list[0])}\n")
        if config['semi_supervised']:
            f.write(f"Semi Supervised Activated, Train {config['semi_ratio']}\n")
        elif config['few_shot']:
            f.write(f"FSL activated, FSL shots {config['shots_per_class']}\n")
        f.write("="*70 + "\nCore Intel Data be:" + "="*30+"\n")
        if len(core_intel):
            for key, value in core_intel.items():
                f.write(f"Key {key} shows: {value}\n")
        f.write("="*70 + "\n\n")
        
        for i, (pred, target, text) in enumerate(zip(*pred_target_text_list)):
            pred_emotion = EMOTION_LIST[pred]
            target_emotion = EMOTION_LIST[target]
            
            f.write(f"[Sample {i+1}]\n")
            f.write(f"[target emotion: {target_emotion} -- pred emotion: {pred_emotion}]")
            f.write(" ✓\n" if pred == target else " ✗\n")
            f.write(f"[Input Text]\n{text}\n")
            f.write("-"*70 + "\n\n")
    
    accuracy = core_intel['test_accuracy']
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Results saved: {output_file}")
    return accuracy

def json_format_record(output_file: str, device, core_intel: Dict[str, Optional[float | int]], config: Dict[str, Optional[str|int]], pred_target_text_list: List[Optional[List[int] | List[str]]]):
    # Build the output structure
    output_file = output_file.replace('txt', 'json')
    output_data = {
        "device": str(device),
        "total_samples": len(pred_target_text_list[0]),
        "parameters": {
            "semi_supervised": str(config['semi_supervised']),
            "few_shot": str(config['few_shot']),
            "shots_per_class": config.get('shots_per_class', None),
            "semi_ratio": config.get('semi_ratio', None)
        },
        "core_intel": core_intel,
        "text_output": list()
    }
    
    # Build text_output list
    for i, (pred, target, text) in enumerate(zip(*pred_target_text_list)):
        pred_emotion = EMOTION_LIST[pred]
        target_emotion = EMOTION_LIST[target]
        is_correct = (pred == target)
        
        sample_dict = {
            "sample_idx": i + 1,
            "target_emotion": target_emotion,
            "pred_emotion": pred_emotion,
            "is_correct": bool(is_correct),
            "text": text
        }
        output_data["text_output"].append(sample_dict)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    accuracy = core_intel.get('test_accuracy', 0.0)
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Results saved: {output_file}")
    return accuracy

def run_inference(checkpoint_dir, use_val_set: bool, core_intel, device='cuda:3', output_file='predictions', store_json: bool=True):
    """运行推理并保存结果"""
    
    print(f"\n{'='*70}")
    print(f"Inference: {os.path.basename(checkpoint_dir)}")
    print(f"{'='*70}")
    
    model, tokenizer, config = load_checkpoint_for_inference(checkpoint_dir, device)
    val_loader, test_loader = prepare_data_loaders(config, tokenizer)
    data_loader = val_loader if use_val_set else test_loader
    
    all_predictions = []
    all_targets = []
    all_texts = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, desc="Inferring")):
            # if idx > 200: break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model.forward({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
            
            preds = torch.argmax(logits, dim=-1)
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            non_match_mask = torch.ones(preds.shape, dtype=bool).numpy()
            # print('non match mask shape', non_match_mask.shape)
            
            all_predictions.extend(preds.cpu().numpy()) # [non_match_mask]
            all_targets.extend(labels.cpu().numpy()) # [non_match_mask]
            # Filter texts using list comprehension since texts is a Python list
            all_texts.extend([text for text, mask in zip(texts, non_match_mask) if mask])
    
    final_output_file = output_file + '.txt'
    accuracy = txt_format_record(output_file=final_output_file, device=device, core_intel = core_intel, config=config, pred_target_text_list=[all_predictions, all_targets, all_texts])
    if store_json:
        final_output_file = output_file+".json"
        jacc = json_format_record(output_file=final_output_file, device=device, core_intel = core_intel, config=config, pred_target_text_list=[all_predictions, all_targets, all_texts])
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return accuracy


def find_experiment_dirs(outputs_root='./outputs'):
    """找到所有包含summary.txt的实验目录"""
    
    experiment_dirs = []
    
    for date_folder in os.listdir(outputs_root):
        date_path = os.path.join(outputs_root, date_folder)
        
        if not os.path.isdir(date_path) or date_folder == 'params_update_check':
            continue
        
        for exp_folder in os.listdir(date_path):
            exp_path = os.path.join(date_path, exp_folder)
            
            if os.path.isdir(exp_path) and os.path.exists(os.path.join(exp_path, 'summary.txt')):
                experiment_dirs.append(exp_path)
    
    print(f"Found {len(experiment_dirs)} completed experiments")
    return experiment_dirs


def get_best_checkpoint(exp_dir):
    """获取实验的最后一个checkpoint"""
    
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        return None
    
    checkpoints = [
        int(d.split("-")[-1])
        for d in os.listdir(checkpoints_dir) 
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoints_dir, d))
    ]
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x)) # os.path.basename(x).split('-')[1]
    return checkpoints[-1]


def batch_inference_all_experiments(outputs_root='./outputs', device='cuda:3', use_val_set=True):
    """批量对所有实验运行推理"""
    
    exp_dirs = find_experiment_dirs(outputs_root)
    exp_dirs = ['outputs/Qwen3_02-11/method_SSP_bs_6_inputdata_ws_prompt_Qwen1d7_SSP_01'] # 'outputs/Llama_02-11/method_FSL_bs_4_inputdata_ws_prompt_Llama3.3_FSL24', 'outputs/Qwen3_02-11/method_FSL_bs_6_inputdata_ws_prompt_Qwen4b_FSL32', 'outputs/Qwen3_02-11/method_SSP_bs_16_inputdata_ws_prompt_Qwen1d7_SSP_03'
    
    print(f"\n{'='*70}")
    print(f"Processing {len(exp_dirs)} experiments on {device}")
    print(f"Dataset: {'Validation' if use_val_set else 'Test'}")
    print(f"{'='*70}\n")
    
    best_accuracy, best_file_name = 0, ''
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        date_name = os.path.basename(os.path.dirname(exp_dir))
        
        checkpoint = get_best_checkpoint(exp_dir)
        
        if not checkpoint:
            print(f"⚠️ No checkpoint: {date_name}/{exp_name}")
            continue
        
        print(f"\n[{date_name}/{exp_name}]")
        
        accuracy = inference_single_experiment(exp_dir, checkpoint, device=device, use_val_set=use_val_set)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_file_name = exp_dir
    
    print(f"\n Best result {best_accuracy}\nFrom file path {best_file_name}")
    
    return best_accuracy, best_file_name


def inference_single_experiment(exp_dir, checkpoint_step, device='cuda:3', use_val_set=True):
    """对单个实验运行推理"""
    
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints', f'checkpoint-{checkpoint_step}')
    core_intel = get_core_intel(exp_dir)
    
    output_file = os.path.join(exp_dir, f"predictions_step{checkpoint_step}_{'val' if use_val_set else 'test'}")
    
    accuracy = run_inference(checkpoint_dir, use_val_set, core_intel, device, output_file)
    
    return accuracy


if __name__ == "__main__":
    # 批量处理所有实验
    results = batch_inference_all_experiments(
        outputs_root='./outputs',
        device='cuda:3',
        use_val_set=True
    )
    
    # 单个实验示例（取消注释使用）
    # accuracy = inference_single_experiment(
    #     exp_dir='outputs/Qwen3_02-11/method_SSP_bs_8_inputdata_ws_prompt_Qwen1d7_SSP_01',
    #     checkpoint_step=3000,
    #     device='cuda:0',
    #     use_val_set=True
    # )

