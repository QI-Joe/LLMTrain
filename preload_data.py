from typing import List, Optional, Dict
import os, pickle, itertools
import threading
from transformers import AutoTokenizer
from collections import defaultdict


def load_empathetic_data(data_dir):
    """
    加载原始EmpatheticDialogues数据 (对齐src/utils/data/loader.py格式)
    
    Returns:
        data_tra, data_val, data_tst: 字典格式
            {
                "context": List[List[List[str]]],  # [样本数][轮数][词]
                "target": List[List[str]],          # [样本数][词]
                "emotion": List[str],               # [样本数]
                "situation": List[List[str]],       # [样本数][词]
                ...
            }
        vocab: Lang对象
    """
    cache_file = f"{data_dir}/dataset_preproc.p"
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"数据文件不存在: {cache_file}\n"
            f"请先运行 src/utils/data/loader.py 生成预处理数据"
        )
    
    print(f"加载数据: {cache_file}")
    with open(cache_file, "rb") as f:
        data_tra, data_val, data_tst, vocab = pickle.load(f)
    
    print(f"训练集: {len(data_tra['emotion'])} 样本")
    print(f"验证集: {len(data_val['emotion'])} 样本")
    print(f"测试集: {len(data_tst['emotion'])} 样本")
    
    return data_tra, data_val, data_tst, vocab

def combine_results(result_list: List[Dict]) -> Dict[str, List[str]]:
    """
    合并多线程处理结果为统一的字典格式
    
    Args:
        result_list: 每个元素包含 {'data': List[Dict], 'max_lens': tuple}
    
    Returns:
        combined_dict: {'ws_prompt': List[str], 'wo_prompt': List[str], 'emotion': List[str]}
        max_ws_len: int
        max_wo_len: int
    """
    combined = defaultdict(list)
    max_ws_len, max_wo_len = 0, 0
    
    for result in result_list:
        if result is None:
            continue
            
        # 提取每条数据
        for item in result['data']:
            combined['ws_prompt'].append(item['ws_prompt'])
            combined['wo_prompt'].append(item['wo_prompt'])
            combined['emotion'].append(item['emotion'])
            combined['ud_idx'].append(item['unique_dialogue_idx'])
            combined['ld_idx'].append(item['local_dialogue_idx'])
        
        # 更新最大长度
        local_ws, local_wo = result['max_lens']
        max_ws_len = max(max_ws_len, local_ws)
        max_wo_len = max(max_wo_len, local_wo)
    
    return dict(combined), max_ws_len, max_wo_len

def save_processed_data(data_dict: Dict[str, List[str]], max_ws_len: int, max_wo_len: int, save_dir: str):
    """
    保存处理后的数据到pickle文件
    
    Args:
        data_dict: 包含'ws_prompt', 'wo_prompt', 'emotion'的字典
        max_ws_len: 带情境的最大prompt长度
        max_wo_len: 不带情境的最大prompt长度
        save_dir: 保存目录 (如 './data/emotion_cls/')
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'processed_data.pkl')
    
    # 打包所有数据
    save_obj = (data_dict, max_ws_len, max_wo_len)
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"数据已保存到: {save_path}")
    print(f"  - 样本数: {len(data_dict['emotion'])}")
    print(f"  - 最大长度: ws={max_ws_len}, wo={max_wo_len}")
    return save_path


def load_processed_data(save_dir: str):
    """
    加载处理后的数据
    
    Returns:
        data_dict: 数据字典
        max_ws_len: 最大带情境长度
        max_wo_len: 最大不带情境长度
    """
    load_path = os.path.join(save_dir, 'processed_data.pkl')
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"未找到预处理数据: {load_path}")
    
    with open(load_path, 'rb') as f:
        data_dict, max_ws_len, max_wo_len = pickle.load(f)
    
    print(f"数据已加载: {load_path}")
    return data_dict, max_ws_len, max_wo_len

def clip_dialogue_window(context_list: Dict[str, List[List[str]]], tokenizer, result_list: List, idx: int, interval: int = 5000):
    """
    put a single dialogue turn into a template
    
    Args:
        context_list: Dict包含'context', 'situation', 'emotion'三个键
        tokenizer: tokenizer实例
        result_list: 共享列表用于存储结果
        idx: 当前线程索引
    """
    system_msg_template = [{
        "role": "system",
        "content": "You are an emotion recognition assistant. Analyze the dialogue and classify the emotion of the user's final response."
    }]
    
    local_max_ws, local_max_wo = 0, 0
    processed_data = []
    unique_dialogue_idx = idx*interval - 1 
    local_dialogue_idx = 0
    
    # 正确的zip三个列表
    for ctx, sit, emo in zip(
        context_list['context'], 
        context_list['situation'], 
        context_list['emotion']
    ):
        sit_text = ' '.join(sit)
        
        # 处理对话历史 - 判断是否为新对话开始
        if len(ctx) <= 1:
            # 新对话开始：只有首轮发言
            past_history, syswords, context = '', '', ' '.join(ctx[0]) if ctx else ''
            unique_dialogue_idx += 1
            local_dialogue_idx = 0  # 首轮编号为0
        else:
            # 对话继续：有多轮历史
            syswords = ' '.join(ctx[-2])
            context = ' '.join(ctx[-1])
            # past_history包含除最后两轮外的所有历史
            past_history = ' '.join([' '.join(sentence) for sentence in ctx[:-2]])
            local_dialogue_idx += 1
        
        # 每次创建新的消息列表，避免修改共享对象
        ws_messages = system_msg_template + [{
            "role": "user",
            "content": f"History: {past_history}\nSituation: {sit_text}\nContext: [Asker: {syswords}] [User: {context}]",
        }]
        
        wo_messages = system_msg_template + [{
            "role": "user",
            "content": f"History: {past_history}\nContext: [Asker: {syswords}] [User: {context}]",
        }]
        
        ws_prompt = tokenizer.apply_chat_template(ws_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        wo_prompt = tokenizer.apply_chat_template(wo_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        processed_data.append({
            'ws_prompt': ws_prompt,
            'wo_prompt': wo_prompt,
            'emotion': emo,
            'unique_dialogue_idx': unique_dialogue_idx,
            'local_dialogue_idx': local_dialogue_idx,
        })
        
        local_max_ws = max(local_max_ws, len(ws_prompt))
        local_max_wo = max(local_max_wo, len(wo_prompt))
    
    # 线程安全：存储到指定索引位置
    result_list[idx] = {
        'data': processed_data,
        'max_lens': (local_max_ws, local_max_wo)
    }
    
    return processed_data, (local_max_ws, local_max_wo)

def union_and_extract_text(datadict: List[List[str]], interval: int = 5000):
    ctx, sit, emotion = list(), list(), list()
    for data in datadict:
        ctx += data['context']
        sit += data['situation']
        emotion += data['emotion'].tolist()
    
    lenctx, lensit, lenemo = len(ctx), len(sit), len(emotion)
    assert lenctx == lensit ==lenemo, f'Expect context, situation, and emotion the same length, got ctx: {lenctx}, sit: {lensit}, emo: {lenemo} instead'
    return [{
        'context': ctx[i:min(i+interval, lenctx)],
        'situation': sit[i:min(i+interval, lenctx)],
        'emotion': emotion[i:min(i+interval, lenctx)],
    }
        for i in range(0, lenctx, interval)]

def data_view(data: Dict[str, List[str]]):
    """View through OLD Version data get from function load_empathetic_data

    Args:
        data (Dict[str, List[str]])
    """
    for i in range(20):
        emotion = data['emotion'][i]
        print(' '.join(list(itertools.chain.from_iterable(data['context'][i]))))
        if len(data['context'][i]) <=1: 
            print(f'Emotion: {emotion}\n\n')
            continue
        sysrespon, user = data['context'][i][-2:]
        print(f"Sys words: {' '.join(sysrespon)}\nUser words: {' '.join(user)}\nSituation: {' '.join(data['situation'][i])}\nEmotion: {emotion} \n\n")
        

if __name__=='__main__':
    data_dir = r'./data'
    save_dir = os.path.join(data_dir, 'emotion_cls')
    data_tra, data_val, data_tst, vocab = load_empathetic_data(data_dir)
    under_processing = True
    interval = 100_000
    
    if under_processing:
        used_data_list = [data_tra, data_val, data_tst]
        
        token_path = os.path.expanduser(r'~/Documents/LLModel/Llama-3.3-8B-Instruct')
        tokenizer = AutoTokenizer.from_pretrained(token_path)
        
        data_dict: List[Dict[str, List]] = union_and_extract_text(used_data_list, interval = interval)
        thread_nums = len(data_dict)
        print(f'Number of Thread is {thread_nums}')
        
        # 创建共享结果列表（预分配空间，线程安全）
        result_list = [None] * thread_nums
        
        thread_list = []
        for i in range(thread_nums):
            # tokenizer是只读操作，不需要deepcopy，可以共享
            thread = threading.Thread(
                target=clip_dialogue_window, 
                args=(data_dict[i], tokenizer, result_list, i, interval)
            )
            thread_list.append(thread)
            
        # 启动所有线程
        for thread in thread_list:
            thread.start()
        
        # 等待所有线程完成
        for thread in thread_list:
            thread.join()
        
        # 合并所有线程结果
        combined_data, max_ws_len, max_wo_len = combine_results(result_list)
        
        print(f"\n处理完成: 共{len(combined_data['emotion'])}条数据")
        print(f"最大prompt长度 - 带情境: {max_ws_len}, 不带情境: {max_wo_len}")
        
        # 保存到pickle文件
        save_path = save_processed_data(combined_data, max_ws_len, max_wo_len, save_dir)
        
    # 验证：尝试加载
    print("\n验证数据加载...")
    loaded_data, loaded_ws, loaded_wo = load_processed_data(save_dir)
    print(f"✓ 加载成功: {len(loaded_data['emotion'])}条数据")
    print(f"最大prompt长度 - 带情境: {loaded_ws}, 不带情境: {loaded_wo}")
    
    for i in range(10):
        print("#"*23, f"Example {i}", "#"*23)
        print(f"with Situation Pormpt: {loaded_data['ws_prompt'][i]}\n\n\nWithout Situation Prompt: {loaded_data['wo_prompt'][i]}Emotion: {loaded_data['emotion'][i]}\nDialogue Index: {loaded_data['ud_idx'][i]}\nLocal Dialogue Index: {loaded_data['ld_idx'][i]}")
        print("#"*50, '\n\n\n')
    print(f"最大prompt长度 - 带情境: {loaded_ws}, 不带情境: {loaded_wo}")
    
    