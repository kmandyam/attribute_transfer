from typing import List, Dict, Tuple
from tools.generate_attributes import retrieve_attribute_vocabulary
from nltk import ngrams

negative_vocab, positive_vocab = retrieve_attribute_vocabulary()

def get_content(sentence: List[str], attribute: str) -> Tuple[List[str], List[str]]:
    """
    Splits a sentence into its content (and attribute markers)
    :param sentence: A list of strings representing words in the original sentence
    :param attribute: A string representing the attribute
    :return: A list of strings representing the content and a list of strings representing
    the attribute markers
    """
    assert attribute in ["positive", "negative"]

    attr_vocab = negative_vocab if attribute is "negative" else positive_vocab
    content, attribute_markers = remove_markers(sentence, attr_vocab)
    return content, attribute_markers

def remove_markers(sentence: List[str], attr_vocab: Dict[str, float]) -> Tuple[List[str], List[str]]:
    # generate all ngrams for the sentence
    grams = []
    for i in range(1, 5):
        i_grams = [" ".join(gram) for gram in ngrams(sentence, i)]
        grams.extend(i_grams)

    # filter ngrams by whether they appear in the attribute_vocab
    candidate_markers = [(gram, attr_vocab[gram]) for gram in grams if gram in attr_vocab]

    # sort attribute markers by score and prepare for deletion
    content = " ".join(sentence)
    candidate_markers.sort(key=lambda x: x[1], reverse=True)
    candidate_markers = [marker for (marker, score) in candidate_markers]

    # delete based on highest score first
    attribute_markers = []
    for marker in candidate_markers:
        if marker in content:
            attribute_markers.append(marker)
            content = content.replace(marker, "")
    return content.split(), attribute_markers
