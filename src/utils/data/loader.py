import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
# from src.utils import config
import torch.utils.data as data
# from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List
from ZGeneration.config_gen import GenTrainingConfig


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0] # ../data/NRCDict.json when run in data_test.ipynb
try:
    stop_words = stopwords.words("english")
except Exception:
    nltk.download("stopwords")
    stop_words = stopwords.words("english")
    print("NLTK stopwords initialized.")
config: GenTrainingConfig = None
SYSTEM_TEMPLATE = "You are the assistant trying to show your empathy to the user during the "\
                    "conversation. Please don't over reply to the user's message (i.e., no need to use so many sentences.). "\
                    "Reply to the user's message as naturally as possible."


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            if i == len(ctx) - 1:
                get_commonsense(comet, item, data_dict)

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab

def load_comet_data(file_name):
    data_dir = config.data_dir
    cache_file = f"{data_dir}/{file_name}"
    print(f"LOADING COMET data of empathetic dialogue: {file_name}")
    with open(cache_file, "rb") as f:
        [comet_tra, comet_val, comet_tst] = pickle.load(f)
    return comet_tra, comet_val, comet_tst

def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Data missed in config part,", data_dir)

    for i in range(10):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader.
    
    Migrated to use LLM tokenizer instead of custom vocab.
    Uses apply_chat_template for dialogue context formatting.
    """

    def __init__(self, data, tokenizer, add_CLS=False):
        """
        Args:
            data: Preprocessed data dict containing context, target, emotion, etc.
            tokenizer: HuggingFace tokenizer from the LLM.
            add_CLS: Whether to prepend CLS token (default False, reserved for future use).
        """
        self.tokenizer = tokenizer
        self.tokenizer.chat_template = tokenizer.chat_template.replace(
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
            ""
        )
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()
        self.add_CLS = add_CLS
        
        # Setup special token IDs from tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
        
        # If add_CLS is True, ensure CLS token exists in tokenizer
        # This should be done externally before creating Dataset:
        # tokenizer.add_special_tokens({"additional_special_tokens": ["[CLS]"]})
        # model.resize_token_embeddings(len(tokenizer))
        if self.add_CLS:
            if "[CLS]" in self.tokenizer.get_vocab():
                self.cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
            else:
                raise ValueError("add_CLS=True but [CLS] token not found in tokenizer. "
                                 "Add it via tokenizer.add_special_tokens() first.")
        
        # Store relations for comet data processing
        self.relations = ["intent", "need", "want", "effect", "react"]
        self.valid_data_type = ["c", "s"]

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        
        # Store raw text data
        item["context_text"] = self.data["context"][index]  # List of word lists
        item["situation_text"] = self.data["situation"][index]  # Word list
        item["target_text"] = self.data["target"][index]  # Word list
        item["emotion_text"] = self.data["emotion"][index]  # Emotion string
        item["emotion_context"] = self.data["emotion_context"][index]  # List of emotion words

        # Sentiment analysis on first context utterance
        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        # Process context using chat template (joined text with speaker roles)
        item["context"], item['labels'], item['prompt_ids'] = self.preprocess_context(item["context_text"], item['target_text'])
        
        # Process target (response to generate)
        item["target"] = self.preprocess_text(item["target_text"], add_eos=True)
        
        # Process emotion
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )

        # Process emotion context (list of emotion words → tokenize as joined text)
        item["emotion_context"] = self.preprocess_text(item["emotion_context"])

        # Commonsense data from utt_cs
        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        # Process commonsense relations
        item["x_intent"] = self.preprocess_commonsense(item["x_intent_txt"])
        item["x_need"] = self.preprocess_commonsense(item["x_need_txt"])
        item["x_want"] = self.preprocess_commonsense(item["x_want_txt"])
        item["x_effect"] = self.preprocess_commonsense(item["x_effect_txt"])
        item["x_react"] = self.preprocess_commonsense(item["x_react_txt"])

        # Process situation
        item["situation"] = self.preprocess_text(item["situation_text"])

        # Process comet data for context and situation
        self.process_comet_data(item, self.data["comet_cxt"][index], data_type="c")
        self.process_comet_data(item, self.data["comet_sit"][index], data_type="s")

        return item

    def preprocess_context(self, context_turns, target_turns: str):
        """
        Convert dialogue context to chat format using apply_chat_template.
        
        Args:
            context_turns: List of word lists, where even indices are user, odd are assistant.
        
        Returns:
            torch.LongTensor of token IDs.
        """
        # Build messages list for chat template
        messages = [
            {
                'role': 'system',
                'content': SYSTEM_TEMPLATE
            }
        ]
        for i, turn in enumerate(context_turns):
            # Convert word list to string
            text = " ".join(turn) if isinstance(turn, list) else turn
            # Even index = user, Odd index = assistant
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})
        
        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Add prompt for next assistant turn
            return_tensors=None  # Return list, we'll convert to tensor
        )
        
        full_text_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": ' '.join(target_turns)}],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        
        ctx_len = len(input_ids)
        labels = [-100] * ctx_len + full_text_ids[ctx_len: ]
        
        # Optionally prepend CLS token
        if self.add_CLS:
            full_text_ids = [self.cls_id] + full_text_ids
            labels = [-100] + labels  # CLS token 本身不需要计算 loss，所以用 -100
            input_ids = [self.cls_id] + input_ids
        
        return torch.LongTensor(full_text_ids), torch.LongTensor(labels), torch.LongTensor(input_ids)

    def preprocess_text(self, word_list, add_eos=False):
        """
        Tokenize a word list (or list of words) into token IDs.
        
        Args:
            word_list: List of words to join and tokenize.
            add_eos: Whether to append EOS token.
        
        Returns:
            torch.LongTensor of token IDs.
        """
        # Join words into text
        text = " ".join(word_list) if isinstance(word_list, list) else word_list
        
        # Tokenize
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Optionally prepend CLS
        if self.add_CLS:
            input_ids = [self.cls_id] + input_ids
        
        # Optionally append EOS
        if add_eos and self.eos_id is not None:
            input_ids = input_ids + [self.eos_id]
        
        return torch.LongTensor(input_ids)

    def preprocess_commonsense(self, cs_data):
        """
        Process commonsense data (list of sentence word-lists).
        
        Args:
            cs_data: List of word lists representing commonsense inferences.
        
        Returns:
            List of torch.LongTensor, one per inference.
        """
        result = []
        for sent in cs_data:
            # Filter out "to" and "none" as in original
            if isinstance(sent, list):
                words = [w for w in sent if w not in ["to", "none"]]
                text = " ".join(words)
            else:
                text = sent
            
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            if self.add_CLS:
                input_ids = [self.cls_id] + input_ids
            
            result.append(torch.LongTensor(input_ids))
        
        return result

    def process_comet_data(self, item, data, data_type):
        """
        Process COMET commonsense data for context or situation.
        
        Args:
            item: The item dict to populate.
            data: COMET data for this sample.
            data_type: "c" for context, "s" for situation.
        """
        for i, r in enumerate(self.relations):
            if data_type == "c":
                # For context, gather relation data from all turns
                r_data = [d[i] for d in data]
            else:
                # For situation, single entry
                r_data = [data[i]]
            
            item[f"{data_type}_{r}_txt"] = r_data
            
            # Process each relation's data
            processed = []
            for rd in r_data:
                if isinstance(rd, list) and len(rd) > 0:
                    # rd is list of word lists (multiple inferences)
                    for sent in rd:
                        if isinstance(sent, list):
                            words = [w for w in sent if w not in ["to", "none"]]
                            text = " ".join(words)
                        else:
                            text = sent
                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        if self.add_CLS:
                            ids = [self.cls_id] + ids
                        processed.append(torch.LongTensor(ids))
                elif isinstance(rd, str):
                    ids = self.tokenizer.encode(rd, add_special_tokens=False)
                    if self.add_CLS:
                        ids = [self.cls_id] + ids
                    processed.append(torch.LongTensor(ids))
            
            # If empty, add a placeholder empty tensor
            if not processed:
                processed.append(torch.LongTensor([]))
            
            item[f"{data_type}_{r}"] = processed

    def preprocess_emo(self, emotion, emo_map):
        """
        Process emotion label into one-hot and label index.
        """
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data, pad_token_id=0):
    """
    Collate function for DataLoader.
    
    Args:
        data: List of items from Dataset.__getitem__
        pad_token_id: Token ID to use for padding (right padding).
    """
    def merge(sequences, pad_id):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.full(
            (len(sequences), max(lengths)), pad_id, dtype=torch.long
        )  ## right padding with pad_token_id
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def left_merge(sequences, pad_id):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded_seqs = torch.full(
            (len(sequences), max_len), pad_id, dtype=torch.long
        )  ## left padding with pad_token_id
        for i, seq in enumerate(sequences):
            length = lengths[i]
            padded_seqs[i, max_len - length:] = seq[:length]
        return padded_seqs, lengths

    def context_merge(sequences, pad_id):
        """Merge list of list of tensors (for commonsense data)."""
        lengths = [[len(seq) for seq in ss] for ss in sequences]
        bz = len(lengths)
        max_len = max([max(s) if s else 1 for s in lengths])
        sent_num = max([len(s) for s in lengths])
        padded_seqs = torch.full((bz, sent_num, max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            for j, ss in enumerate(seq):
                end = lengths[i][j]
                if end > 0:
                    padded_seqs[i, j, :end] = ss[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input - context is now a single tensor per sample (from apply_chat_template)
    input_toks, labels = item_info['context'], item_info['labels']
    input_batch, input_lengths = merge(input_toks, pad_token_id)
    attention_mask = input_batch != pad_token_id
    labels, _ = merge(labels, -100)
    
    prompt_toks = item_info['prompt_ids']
    prompt_batch, prompt_lengths = left_merge(prompt_toks, pad_token_id)
    prompt_mask = prompt_batch!=pad_token_id

    ## emotion context
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"], pad_token_id)

    ## Target
    target_batch, target_lengths = merge(item_info["target"], pad_token_id)
    
    ## Situation
    situation_batch, situation_lengths = merge(item_info["situation"], pad_token_id)
    situation_mask = situation_batch!=pad_token_id

    input_batch = input_batch.to(config.device)
    attention_mask = attention_mask.to(config.device)
    target_batch = target_batch.to(config.device)
    situation_batch = situation_batch.to(config.device)
    prompt_batch = prompt_batch.to(config.device)
    prompt_mask = prompt_mask.to(config.device)
    labels = labels.to(config.device)    
    situation_mask = situation_mask.to(config.device)
    
    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = input_lengths
    d['labels'] = labels
    d["attention_mask"] = attention_mask
    d['prompt_ids'], d['prompt_mask'] = prompt_batch, prompt_mask # modify;
    
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths).to(config.device)
    d["emotion_context_batch"] = emotion_batch.to(config.device) 

    d["situation_batch"] = situation_batch
    d["situation_lengths"] = torch.LongTensor(situation_lengths).to(config.device)
    d["situation_attn_mask"] = situation_mask

    ## program (emotion)
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ## text
    d["input_txt"] = item_info["context_text"]
    d["target_text"] = [' '.join(stcs) for stcs in item_info["target_text"]] # modify;
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    # comet data
    relations = ["intent", "need", "want", "effect", "react"]
    valid_data_type = ["c", "s"]
    for prefix in valid_data_type:
        for r in relations:
            key = f"{prefix}_{r}"
            pad_batch, _ = context_merge(item_info[key], pad_token_id)
            pad_batch = pad_batch.to(config.device)
            d[key] = pad_batch
            d[f"{key}_txt"] = item_info[f"{key}_txt"]

    return d

def load_idf(load_path="data/data/updated_vocab_idf.json"):
    with open(load_path, 'r') as f:
        print("LOADING vocabulary idf")
        idf_json = json.load(f)
    max_idf = 0.
    mean_idf = 0.0 
    min_idf = 99.0
    for key in idf_json:
        idf = idf_json[key]
        if max_idf < idf:
            max_idf = idf 
        if min_idf > idf:
            min_idf = idf 
        mean_idf += idf 
    print(f"Max idf: {max_idf}, Mean idf: {mean_idf / len(idf_json)}, Min idf: {min_idf}")
    return idf_json 

from typing import Optional, List, Dict
import copy
def quick_cut_down(train: Dict[str, Optional[List[List] | List]], ratio: float):
    len_train = len(train["target"]) / 0.8
    cut_num = int(len_train * ratio) if ratio < 0.79 else int(len_train * 0.79)
    
    while len(train['context'][cut_num]) > 1:
        cut_num += 1
    print(f"Cutting down training data from {len_train} to {cut_num}")
    new_train = copy.deepcopy(train)
    for key in train.keys():
        new_train[key] = train[key][:cut_num]
    return new_train


def prepare_data_seq(tokenizer, input_config_setting: GenTrainingConfig):
    from functools import partial
    global config 
    config = input_config_setting
    
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    pairs_tra = quick_cut_down(pairs_tra, config.semi_ratio)

    logging.info("Vocab  {} ".format(vocab.n_words))
    
    # Create collate_fn with tokenizer's pad_token_id
    pad_id = tokenizer.pad_token_id
    collate_with_pad = partial(collate_fn, pad_token_id=pad_id)

    dataset_train = Dataset(pairs_tra, tokenizer)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_with_pad,
    )

    dataset_valid = Dataset(pairs_val, tokenizer)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_with_pad,
    )
    dataset_test = Dataset(pairs_tst, tokenizer)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=collate_with_pad
    )
    # save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        tokenizer,  # Return tokenizer instead of vocab
        len(dataset_train.emo_map),
        dataset_train,
    )
