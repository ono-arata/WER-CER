from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from janome.tokenizer import Tokenizer

def calculate_bleu(hypothesis, reference):
    t = Tokenizer()
    hypothesis_tokens = [token.surface for token in t.tokenize(hypothesis)]
    reference_tokens = [[token.surface for token in t.tokenize(reference)]]
    score = sentence_bleu(reference_tokens, hypothesis_tokens)
    return score

def calculate_bertscore(hypothesis, reference):
    P, R, F1 = score([hypothesis], [reference], lang="ja")
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

def calculate_rouge(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


with open("hypothesis.txt", 'r', encoding='utf-8') as file:
    hypothesis = file.read()
with open("reference.txt", 'r', encoding='utf-8') as file:
    reference = file.read()
bleu_score = calculate_bleu(hypothesis, reference)
print(f"BLEU Score: {bleu_score}")
rouge_scores = calculate_rouge(hypothesis, reference)
print("ROUGE Scores:", rouge_scores)
bert_scores = calculate_bertscore(hypothesis, reference)
print("BERTScores:", bert_scores)
