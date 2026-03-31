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


