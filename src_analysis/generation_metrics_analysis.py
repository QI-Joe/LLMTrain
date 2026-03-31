# # Generation Task Metric Analysis
# This notebook analyzes the evaluation results from the Generation Task. 
# It focuses on comparing the performance of the model on the **First Turn** vs **Last Turn** of each dialogue.
# 
# ## Requirements
# 1.  Compute average metrics of **FIRST-TURN** and **LAST-TURN** for each dialogue.
# 2.  Count how many dialogues have **FIRST-TURN** results better than **LAST-TURN** results.
# 

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
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), 'outputs'))

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
    # "llama3.1Gen_03-18",
    "llama3.1Gen_03-30",
    # '../'
    # 'raw_model_Gen_Qwen3_3-4',
    
    # 'eval_results',
]

outer_files = [
    "../"
]

jsonl_files = []

print(f"Searching in: {BASE_OUTPUT_DIR}")
for folder in TARGET_FOLDERS:
    # Construct search path: outputs/{folder_name}/**/eval_logs/*.jsonl
    search_path = os.path.join(BASE_OUTPUT_DIR, folder, '**', 'eval_logs', '*.jsonl')
    found_files = glob.glob(search_path, recursive=True)
    
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
jsonl_files = [jf for jf in jsonl_files if re.search(repattern, jf, flags=re.IGNORECASE) or re.search(repattern2, jf, flags=re.IGNORECASE)]

jsonl_files.extend(outside_files)
print(f"[After] Found {len(jsonl_files)} log files.")
for f in jsonl_files:
    print(f" - {f}")

# %%
from collections import Counter
import numpy as np

# --- 1. The Baidu NLP Function ---
def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    
    for seq in seqs:
        # Intra: Diversity within a single sentence
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        # Update global counts for Inter metrics
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    # Inter: Diversity across the whole set of sentences (the whole dialogue)
    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

# %%
# 2. Load Data into DataFrame
data_records = []

for file_path in jsonl_files:
    # Attempt to extract experiment/run name from path
    # Path format: .../outputs/{Experiment}/{Run}/eval_logs/...
    rel_path = os.path.relpath(file_path, BASE_OUTPUT_DIR)
    parts = rel_path.split(os.sep)
    print(f"[parts] the output parts is: {parts}")
    
    # Heuristic for experiment name
    exp_name = "check_logs" # Default
    if len(parts) >= 2 and len(parts)<4:
        exp_name = f"{parts[0]}/{parts[1]}"
    elif len(parts) >=4:
        pattern = r"^method_SSP_bs_[0-9]_inputdata_input_text_(.+)$"
        match, captured = re.match(pattern, parts[1]), 'na'
        if match:
            captured = match.group(1)
        exp_name = f"{captured}/{parts[-1]}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                # Flatten metrics for easier pandas use
                metrics = record.pop('metrics', {})
                for k, v in metrics.items():
                    record[f'metric_{k}'] = v
                
                # Add metadata
                record['source_file'] = file_path
                record['experiment'] = exp_name
                
                if 'epoch' not in record:
                    record['epoch'] = -1
                
                # Ensure types
                record['ud_idx'] = str(record.get('ud_idx', 'unknown'))
                record['ld_idx'] = int(record.get('ld_idx', -1))
                
                data_records.append(record)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

df = pd.DataFrame(data_records)
print(f"Loaded {len(df)} records.")
if not df.empty:
    print(df.head())
else:
    print("No data found. Please ensure 'eval_logs' exist and contain valid jsonl files.")


# %% [markdown]
# ### Specificlly designed for all/avg PPL computation

# %%
# 2. Load Data into DataFrame
grouped_data = dict()

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
        part_name1 = '_'.join(parts[1].split('_')[-2:])
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
    'Llama3.1_FSL16/eval_results_FSL_4shots_last_test': 0,
    'Llama3.1_FSL16/eval_results_FSL_16shots_last_test': 0,
    'Llama3.1_SSL02/eval_results_FSL_4shots_last_test': 0,
    'Llama31_SSL01/eval_results_FSL_4shots_last_test': 0,
    'Llama3.1_SSL02/eval_results_epoch_test_2_rank_0': 0,
    'Llama3.1_SSL02/eval_results_epoch_val_2_rank_0': 0,
}

for modelres in grouped_data:
    record_metrics = dict()
    ppl_list = list()
    inside_list = list()
    
    for idx, singleres in enumerate(grouped_data[modelres]):
        target_text = singleres.get('reference', '')
        if target_text == '':
            target_text = singleres.get('target', '')
        metrics = singleres.get('metrics', {})
        if not len(metrics) or len(target_text)<2: 
            print(f"[idx] {idx}th message have none metrics or short target text {target_text}")
            continue
        
        for key, value in metrics.items():
            if value == np.inf or value == -np.inf or np.isnan(value):
                continue
            if key == 'ppl':
                ppl_list.append(value)
            if key not in record_metrics:
                record_metrics[key] = (value, 1)
            else:
                current_sum, current_count = record_metrics[key]
                record_metrics[key] = (current_sum + value, current_count + 1)
    ppl_list = sorted(ppl_list, reverse=True)
    if modelres not in top_k_dict: top_k_dict[modelres] = 0
    ppl_adjusted_list = ppl_list[top_k_dict[modelres]:]
    print(f"Deleted PPL values are {ppl_list[:top_k_dict[modelres]]}")
    record_metrics['adjust_ppl'] = (sum(ppl_adjusted_list), len(ppl_adjusted_list))
    
    print(f"Model: {modelres}")
    for key, (total, count) in record_metrics.items():
        avg_value = total / count if count > 0 else 0
        print(f"  {key}: {avg_value:.4f} (based on {count} samples)")
    
    print('\n\n')
