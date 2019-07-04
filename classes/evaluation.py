# functions dedicated to reading testing data and evaluating outputs
# main interface involves a function which, when given a model and a file
# evaluates the BLEU score

from typing import List, Dict
import os

# functions to write
# TODO: run model predictor on each line of file
# TODO: aggregate outputs and log them
# TODO: calculate NLTK BLEU score
# TODO: calculate BLEU score the way Li et al. did it

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


test_file = os.path.join(os.path.dirname(__file__), '../data/reference.neg.pos')
print(read_test_file(test_file))


