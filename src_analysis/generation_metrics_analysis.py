# %%
import os
import glob, re
import json, nltk
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# 1. Setup Path
# We search for 'eval_results*.jsonl' in the outputs directory
# Adjust this base path if needed
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd())) # 'llama3-8B')

# Target Folders as requested
TARGET_FOLDERS = [
    # 'Qwen3_02-15',
    # 'Qwen3Gen_03-26',
    # 'raw_model_Gen_llama3.1_2-24',
    # 'raw_model_Gen_Qwen3_2-24',
    # 'raw_model_Gen_llama3.1_2-25',
    # 'raw_model_Gen_Qwen3_2-25',
    # 'raw_model_Gen_Qwen3_2-26',
    # "Qwen3Gen_03-05",
    # "llama3.1Gen_02-24",
    # "llama3.1Gen_03-08",
    # "llama3.1Gen_03-26",
    # "llama3.1Gen_03-30",
    # "llama3.1Gen_03-31",
    # '../'
    # 'raw_model_Gen_Qwen3_3-4',
    # 'llama3-8B/llama3_iamm_Only_Llama_Clean_optimizer_01_ED_2e-05_10',
    # 'llama3-8B/llama3_iamm_Only_Llama_Clean_optimizer_03_ED_2e-05_30',
    # 'llama3-8B/llama3_iamm_Only_Llama_Clean_optimizer_05_ED_2e-05_50',
    # 'Qwen3-4B/Qwen3_iamm_Only_Qwen3_Clean_optimizer_01_ED_2e-05_10',
    # 'Qwen3-4B/Qwen3_iamm_Only_Qwen3_Clean_optimizer_03_ED_2e-05_30',
    
    
    # 'LM_llama3_8B/llama3_lora_empathy_2e-05_13',
    # 'LM_llama3_8B/llama3_lora_empathy_2e-05_15',
    # 'LM_llama3_8B/llama3_lora_empathy_2e-05_17',
    
    # 'LM_Llama3.2-3B/llama32_lora_empathy_2e-05_10',
    # 'LM_llama-3.2-1B/llama32_lora_empathy_2e-05_10',
    # 'LM_llama3_8B/llama3_Cancel_Loss_Explict_KV_cancel_seed_set_2e-05_10',
    # 'LM_llama3_8B/llama3_Cancel_Explict_block_no_grad_2e-05_10',
    # 'LM_llama3_8B/llama3_IAMM_Small_Explict005_2e-05_10',
    
    'LM_llama3_8B/llama3_Cancel_Explict_Block_All_2e-05_10',
    'LM_llama3_8B/llama3_Cancel_Explict_no_grad_all_2e-05_10',
    
    
    
    # 'LM_llama-3.2-1B/llama32_lora_empathy_2e-05_10',
    
    # 'LM_llama3_8B/llama3_Datacollector_dataset_2e-05_10',
    # 'LM_llama3_8B/llama3_IAMM_Explicit_2e-05_10',
    # 'LM_llama3_8B/llama3_Cancel_Explicit_2e-05_10',
    # 'LM_llama3_8B/llama3_Cancel_Loss_Explict_2e-05_10',
    
    # 'llama3-8B/llama3_iamm_Explicity_IAMM_ED_2e-05_10',
    # 'llama3-8B/llama3_iamm_Explicity_IAMM_ED_2e-05_20',
    # 'LM_Llama3.2-3B/llama32_lora_empathy_2e-05_10',
    # 'LM_llama-3.2-1B/llama32_lora_empathy_2e-05_10'
    
    
    # 'eval_results',
]

outer_files = [
    "../"
]

jsonl_files = []

print(f"Searching in: {BASE_OUTPUT_DIR}")
for folder in TARGET_FOLDERS:
    # Construct search path: outputs/{folder_name}/**/eval_logs/*.jsonl
    search_path = os.path.join(BASE_OUTPUT_DIR, '..', folder, '*.jsonl') # '**', 'eval_logs', '*.jsonl')
    found_files = glob.glob(search_path, recursive=True)
    print(search_path)
    
    # Fallback search if 'eval_logs' not found directly inside (e.g. might be simpler structure)
    if not found_files:
         search_path = os.path.join(BASE_OUTPUT_DIR, folder, '**', '*.jsonl')
         found_files = glob.glob(search_path, recursive=True)
         
    jsonl_files.extend(found_files)

outside_files = list()
for outsider in outer_files:
    search_p = os.path.join(BASE_OUTPUT_DIR, outsider, '*.jsonl')
    found_files = glob.glob(search_p, recursive=False)

    outside_files.extend(found_files)

print(f"[Total] Found {len(jsonl_files)} log files.")
for f in jsonl_files:
    print(f"[Total] - {f}")
    
repattern, repattern2 = r"\w+_last_(test|val)", r"eval_results?_\w+_test"
# jsonl_files = [jf for jf in jsonl_files if re.search(repattern, jf, flags=re.IGNORECASE) or re.search(repattern2, jf, flags=re.IGNORECASE)]

# jsonl_files.extend(outside_files)
print(f"[After] Found {len(jsonl_files)} log files.")
for f in jsonl_files:
    print(f" - {f}")


# %%
# 2. Load Data into DataFrame
grouped_data = dict()
from metrics_func import calculate_bleu
for file_path in jsonl_files:
    # Attempt to extract experiment/run name from path
    # Path format: .../outputs/{Experiment}/{Run}/eval_logs/...
    rel_path = os.path.relpath(file_path, BASE_OUTPUT_DIR)
    parts = rel_path.split(os.sep)
    inside_staits_records = list()
    
    print('rel_path:', rel_path, 'parts:', parts)
    
    # Heuristic for experiment name
    exp_name = "check_logs" # Default
    if len(parts) >= 2 and len(parts)<4:
        part_name1 ='_'.join(parts[1].split('_')[-5:]) # parts[0] # 
        exp_name = f"{parts[0]}/{parts[1]}"
        key_name = f"{part_name1}/{parts[-1].split('.')[0]}"
    elif len(parts) >=4:
        pattern = r"^method_SSP_bs_[0-9]_inputdata_input_text_(.+)$"
        match, captured = re.match(pattern, parts[1]), 'na'
        if match:
            captured = match.group(1)
        exp_name = f"{captured}/{parts[-1]}"
        key_name = f"{captured}/{parts[-1].split('.')[0]}"
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                # Add metadata
                record['source_file'] = file_path
                record['experiment'] = exp_name
                
                if 'epoch' not in record:
                    record['epoch'] = -1
                
                # Ensure types
                record['ud_idx'] = str(record.get('ud_idx', 'unknown'))
                record['ld_idx'] = int(record.get('ld_idx', -1))
                
                inside_staits_records.append(record)
            grouped_data[key_name] = inside_staits_records
            print(f'\n\nkey name: {key_name}')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# %%
for key in grouped_data:
    print(f"Experiment: {key}, Number of Records: {len(grouped_data[key])}")

# %%
top_k_dict = {
}

for modelres in grouped_data:
    record_metrics = dict()
    ppl_list = list()
    inside_list = list()
    top_k = 0
    
    for idx, singleres in enumerate(grouped_data[modelres]):
        target_text = singleres.get('reference', '')
        if target_text == '':
            target_text = singleres.get('target', '')
        metrics = singleres.get('metrics', {})
        if len(target_text)<2: 
            # print(f"[Failed target text] at idx {idx} with words {target_text} and metrics length large than 0 {len(metrics)}")
            top_k+=1
        
        for key, value in metrics.items():
            if value == np.inf or value == -np.inf or np.isnan(value):
                print(f"[Inf Warning] The result has -inf/None value {value} at idx {idx} ")
                continue
            if key == 'ppl' : # and not (idx == 2793 or idx == 3533)
                ppl_list.append(value)
            if key not in record_metrics:
                record_metrics[key] = (value, 1)
            else:
                current_sum, current_count = record_metrics[key]
                record_metrics[key] = (current_sum + value, current_count + 1)
    ppl_list = sorted(ppl_list, reverse=True)
    ppl_adjusted_list = ppl_list[top_k:]
    # print(f"Deleted PPL values are {ppl_list[:top_k_dict[modelres]]}")
    record_metrics['adjust_ppl'] = (sum(ppl_adjusted_list), len(ppl_adjusted_list))
    
    print(f"Model: {modelres}")
    for key, (total, count) in record_metrics.items():
        avg_value = total / count if count > 0 else 0
        print(f"  {key}: {avg_value:.4f} (based on {count} samples)")
    
    print('\n\n')



import sys; sys.path.append(os.path.join(os.getcwd(), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics_func import calc_distinct, calc_distinct_tokenizer

tokenizer = AutoTokenizer.from_pretrained("../../LLModel/llama3.1-8B-Instruct", cache_dir='./llama3-8B/', force_download=False)
print("=== Corpus-level Dist-1 and Dist-2 Scores ===")

for model_name, records in grouped_data.items():
    # Extract all generated outputs for this model
    candidates = [record.get("generated", "") for record in records if "generated" in record]
    
    if not candidates:
        print(f"Model: {model_name} - No generated text found.")
        continue
        
    print(f"Model: {model_name}")
    # Compute dist-1 and dist-2 using the imported function from metrics_func
    dist_scores = calc_distinct(candidates, tokenizer, print_score=False)
    print(f"[tokenizer vocab] tokenizer vocab size: {len(tokenizer)}")
    # dist_tokenizer = calc_distinct_tokenizer(candidates, tokenizer, print_score=False)
    
    print(f"  Dist-1: {dist_scores[0]*100:.4f}")
    print(f"  Dist-2: {dist_scores[1]*100:.4f}")
    # print(f"  Dist-1 (Tokenizer): {dist_tokenizer[0]:.4f}")
    # print(f"  Dist-2 (Tokenizer): {dist_tokenizer[1]:.4f}")
    print("-" * 30)
# %%
