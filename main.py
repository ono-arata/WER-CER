from flask import Flask, request, jsonify
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from janome.tokenizer import Tokenizer

app = Flask(__name__)
@app.route('/bleu_scores', methods=['POST'])
def calculate_bleu():
    data = request.get_json()
    hypothesis = data['hypothesis']
    reference = data['reference']
    t = Tokenizer()
    hypothesis_tokens = [token.surface for token in t.tokenize(hypothesis)]
    reference_tokens = [[token.surface for token in t.tokenize(reference)]]
    return jsonify({"bleu_score": sentence_bleu(reference_tokens, hypothesis_tokens)})

@app.route('/bert_scores', methods=['POST'])
def calculate_bertscore():
    data = request.get_json()
    hypothesis = data['hypothesis']
    reference = data['reference']
    P, R, F1 = score([hypothesis], [reference], lang="ja")
    return jsonify({"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()})

@app.route('/rouge_scores', methods=['POST'])
def calculate_rouge():
    data = request.get_json()
    hypothesis = data['hypothesis']
    reference = data['reference']
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, hypothesis)['rougeL']
    precision = score[0]
    recall = score[1]
    f1_score = score[2]
    response ={"precision": precision,
                "recall": recall,
                "f1_score": f1_score}
    
    return jsonify(response)

@app.route('/scores', methods=['POST'])
def scores():
    data = request.get_json()
    hypothesis = data['hypothesis']
    reference = data['reference']
    return jsonify({
        "BLEU": calculate_bleu(hypothesis, reference),
        "BERT": calculate_bertscore(hypothesis, reference),
        "ROUGE": calculate_rouge(hypothesis, reference)
    })

if __name__ == '__main__':
    app.run(port=5009)
