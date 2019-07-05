# functions dedicated to reading testing data and evaluating outputs
# main interface involves a function which, when given a model and a file
# evaluates the BLEU score

from typing import List, Dict
import os
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from nltk.translate.bleu_score import corpus_bleu

# functions to write
# TODO: aggregate outputs and log them
# TODO: calculate NLTK BLEU score

# read data from file and return list of data points
def read_test_file(test_file_name: str) -> List[Dict[str, str]]:
    assert os.path.exists(test_file_name)
    test_data = []
    with open(test_file_name) as f:
        for line in f:
            sentences = line.strip().split("\t")
            test_datum = {"source": sentences[0],
                          "target": sentences[1]}
            test_data.append(test_datum)
    return test_data

# run model predictor on each line of file
def predict_outputs(model: Model,
                    test_data: List[Dict[str, str]],
                    predictor: Predictor,
                    tgt_attr: str) -> List[Dict[str, str]]:
    assert tgt_attr in ["positive", "negative"]
    outputs = []
    for datum in test_data:
        predictions = predictor.predict(datum["source"], tgt_attr)['predictions']
        tokens = [model.vocab.get_token_from_index(i, 'tokens') for i in predictions]
        predicted_sentence = " ".join(tokens)
        evaluation = {
            "input": datum["source"],
            "prediction": predicted_sentence,
            "gold": datum["target"]
        }
        outputs.append(evaluation)
    return outputs

def calculate_bleu(predictions: List[Dict[str, str]]):
    references = [[d["gold"]] for d in predictions]
    hypotheses = [d["prediction"] for d in predictions]

    corpus_score = corpus_bleu(references, hypotheses, [0.25, 0.25, 0.25, 0.25])
    return corpus_score
