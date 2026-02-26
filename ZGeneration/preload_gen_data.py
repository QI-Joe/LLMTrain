from typing import List, Optional, Dict
import os
import pickle
import threading
import sys
from collections import defaultdict
from transformers import AutoTokenizer

# 添加父目录到路径以导入 parallel modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preload_data import load_empathetic_data

def clip_gen_dialogue(context_list: Dict[str, List[List[str]]], tokenizer, result_list: List, idx: int, interval: int = 5000):
    """
    Process dialogue turns into Generation Task format.
    Format:
      Input:
        Situation: <text>
        Context: <text>
      Target:
        <target_text>
    """
    local_max_len = 0
    processed_data = []
    
    # Calculate starting indices
    unique_dialogue_idx = idx * interval - 1 
    local_dialogue_idx = 0
    
    # Iterate through data
    # context_list keys: 'context', 'situation', 'emotion', 'target'
    # Note: 'target' is available in the raw data from load_empathetic_data
    
    for ctx, sit, emo, trg in zip(
        context_list['context'], 
        context_list['situation'], 
        context_list['emotion'],
        context_list['target']
    ):
        # 1. Prepare Text Parts
        sit_text = ' '.join(sit)
        target_str = ' '.join(trg)
        
        # 2. Construct Prompt (Messages Format)
        # Based on Generation_Data_adjust.md
        
        system_msg = {
            "role": "system",
            "content": "U r an empathetic assistant. You need to understand the user's situation, feelings and respond supportively."
        }
        
        messages = [system_msg]
        
        # Process Context (History)
        # EmpatheticDialogues context is usually [User, Assistant, User, Assistant...]
        for i, turn in enumerate(ctx):
            role = "user" if i % 2 == 0 else "assistant"
            content = ' '.join(turn)
            
            # Inject Situation into the first user turn
            if i == 0:
                content = f"Situation: {sit_text}\n\nUser Word: {content}"
                
            messages.append({"role": role, "content": content})
            
        # Add Target (as the final Assistant response for training)
        messages.append({"role": "assistant", "content": target_str})
        
        # We store the structured 'messages' list instead of a raw string.
        # The data loader will handle apply_chat_template.
        # Although the key is 'input_text' for compatibility, it now holds a List[Dict].
        input_data = messages
        
        # 3. Update indices logic (Keeping logic from preload_data.py to maintain block mapping)
        if len(ctx) <= 1:
            unique_dialogue_idx += 1
            local_dialogue_idx = 0
        else:
            local_dialogue_idx += 1
            
        processed_data.append({
            'input_text': input_data,     # Now contains messages list
            'target_text': target_str,    # Keep raw target for reference/metrics if needed
            'emotion': emo,
            'unique_dialogue_idx': unique_dialogue_idx,
            'local_dialogue_idx': local_dialogue_idx,
        })
        
        # Estimate length (sum of content strings)
        current_len = sum(len(m['content']) for m in messages)
        local_max_len = max(local_max_len, current_len)
    
    # Store result
    result_list[idx] = {
        'data': processed_data,
        'max_len': local_max_len
    }

def combine_gen_results(result_list: List[Dict]) -> Dict[str, List[str]]:
    """
    Combine results from threads into a single dict.
    Adapted for Generation keys.
    """
    combined = defaultdict(list)
    max_len = 0
    
    for result in result_list:
        if result is None:
            continue
            
        for item in result['data']:
            combined['input_text'].append(item['input_text'])
            combined['target_text'].append(item['target_text'])
            combined['emotion'].append(item['emotion'])
            combined['ud_idx'].append(item['unique_dialogue_idx'])
            combined['ld_idx'].append(item['local_dialogue_idx'])
        
        max_len = max(max_len, result['max_len'])
            
    return dict(combined), max_len

def save_gen_processed_data(data_dict: Dict, max_len: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'processed_gen_data.pkl')
    
    save_obj = (data_dict, max_len)
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Generation Data saved to: {save_path}")
    print(f"  - Samples: {len(data_dict['emotion'])}")
    print(f"  - Max Input Length (approx): {max_len}")
    return save_path

def union_and_extract_gen(datadict: List[Dict], interval: int = 5000):
    """
    Extract context, situation, target, emotion and chunk them.
    Must include 'target' which was missing in original pre_loader's specific function.
    """
    ctx, sit, emo, trg = [], [], [], []
    for data in datadict:
        ctx += data['context']
        sit += data['situation']
        emo += data['emotion'].tolist() if hasattr(data['emotion'], 'tolist') else data['emotion'] # Handle numpy/list diffs
        trg += data['target']
    
    len_all = len(ctx)
    assert len_all == len(sit) == len(emo) == len(trg), "Data Length Mismatch!"
    
    return [{
        'context': ctx[i:min(i+interval, len_all)],
        'situation': sit[i:min(i+interval, len_all)],
        'emotion': emo[i:min(i+interval, len_all)],
        'target': trg[i:min(i+interval, len_all)]
    } for i in range(0, len_all, interval)]

if __name__ == "__main__":
    # Adjust paths as needed
    # Assuming run from root or ZGeneration folder. 
    # If run from ZGeneration, .. is root.
    # Root contains 'data/'.
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    SAVE_DIR = os.path.join(DATA_DIR, 'gen_task')
    
    print(f"Loading raw data from: {DATA_DIR}")
    data_tra, data_val, data_tst, vocab = load_empathetic_data(DATA_DIR)
    
    used_data_list = [data_tra, data_val, data_tst]
    
    # We don't strictly need the tokenizer for text processing unless calculating token length accurately.
    # We'll pass None for tokenizer if we just want text construction.
    tokenizer = None 
    
    interval = 100_000 
    data_chunks = union_and_extract_gen(used_data_list, interval=interval)
    
    thread_nums = len(data_chunks)
    print(f"Processing in {thread_nums} threads...")
    
    result_list = [None] * thread_nums
    thread_list = []
    
    for i in range(thread_nums):
        thread = threading.Thread(
            target=clip_gen_dialogue,
            args=(data_chunks[i], tokenizer, result_list, i, interval)
        )
        thread_list.append(thread)
        thread.start()
        
    for thread in thread_list:
        thread.join()
        
    combined_data, max_len = combine_gen_results(result_list)
    
    # save_gen_processed_data(combined_data, max_len, SAVE_DIR)
    
    # Validation
    print("\nValidating 3 Samples:")
    for i in range(3):
        print(f"--- Sample {i} ---")
        print(f"INPUT:\n{combined_data['input_text'][i]}")
        print(f"TARGET:\n{combined_data['target_text'][i]}")
        print(f"EMOTION: {combined_data['emotion'][i]}")
        print(f"UD_IDX: {combined_data['ud_idx'][i]}")
        print(f"LD_IDX: {combined_data['ld_idx'][i]}")
        print("-" * 30)



# system_content = f"""You are an empathetic assistant. Your task is to understand the user's emotion and feelings, and respond supportively. <background_info>
#         <situation>{sit_text}</situation></background_info> Please respond to the user's last message based on this background."""

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))