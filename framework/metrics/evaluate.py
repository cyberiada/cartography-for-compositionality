from .bleu import compute_bleu
from typing import List
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def evaluate_bleu(ref: List[str], hyp: List[str]) -> List[float]:
    bleus = []
    for i in range(len(ref)):
        ref_i = [ref[i].split()]
        hyp_i = hyp[i].split()
        
        metrics = sentence_bleu(ref_i, hyp_i, smoothing_function=SmoothingFunction().method4, auto_reweigh=True)
        bleus.append(metrics)

    return bleus