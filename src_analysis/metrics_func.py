import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(reference: str, hypothesis: str):
    ref_tok = nltk.word_tokenize(reference)
    hyp_tok = nltk.word_tokenize(hypothesis)
    
    smooth_func = SmoothingFunction().method1
    # Function 1 is a strict method
    
    bleu1 = sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smooth_func)
    bleu2 = sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_func)
    
    return bleu1, bleu2

def calc_distinct_n(n, candidates, tokenizer, print_score: bool = True):
    ngrams = set()
    total = 0
    
    for candidate in candidates:
        tokens = tokenizer.tokenize(candidate) if hasattr(tokenizer, 'tokenize') else candidate.strip().split()
        tokens = [token.replace('Ġ', '').lower() for token in tokens if token.strip()]
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.add(ngram)
            total += 1
    
    print(f'[N{n}-gram] Unique n-grams: {len(ngrams)}, Total n-grams: {total}')
    score = len(ngrams) / total if total > 0 else 0.0

    if print_score:
        print(f"***** Distinct-{n}: {score*100:.4f} *****")

    return score

def calc_distinct(candidates, tokenizer, print_score: bool = False):
    scores = []
    for i in range(1, 3):
        score = calc_distinct_n(i, candidates, tokenizer, print_score=print_score)
        scores.append(score)

    return scores


def calc_distinct_n_tokenizer(n, candidates, tokenizer, print_score: bool = True):
    ngrams = set()
    
    for candidate in candidates:
        tokens = tokenizer.tokenize(candidate) if hasattr(tokenizer, 'tokenize') else candidate.strip().split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.add(ngram)
            
    vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else getattr(tokenizer, 'vocab_size', 0)
    denominator = float(vocab_size ** n)
    
    score = len(ngrams) / denominator if denominator > 0 else 0.0

    if print_score:
        print(f"***** Tokenizer Distinct-{n}: {score*100:.4f} *****")

    return score

def calc_distinct_tokenizer(candidates, tokenizer, print_score: bool = False):
    scores = []
    for i in range(1, 3):
        score = calc_distinct_n_tokenizer(i, candidates, tokenizer, print_score=print_score)
        scores.append(score)

    return scores


