from typing import List, Dict, Tuple
from tools.generate_attributes import retrieve_attribute_vocabulary
from nltk import ngrams
import random

random.seed(1)
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
    denoised_markers = denoise_markers(attribute_markers, attr_vocab)
    return content, denoised_markers

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

def denoise_markers(markers: List[str], attr_vocab: Dict[str, float]) -> List[str]:
    """
    For each attribute marker, replace with another attribute marker of the same attribute
    and word level edit distance 1 if it exists, with probability 0.1
    :param markers: A list of attribute markers
    :return: A list of attribute markers with noise applied
    """
    for i, marker in enumerate(markers):
        # with probability 0.1
        if random.random() < 0.1:
            # if there exists an attribute marker of edit distance 1, swap it

def get_denoised_marker(marker: str, attr_vocab: Dict[str, float]) -> str:
    # naively, we could loop thorugh all the attribute markers
    # collect all the ones with edit distance 1
    # and then choose the highest one
    # it's preprocessing, so maybe it doesn't matter how fast?

    # lowercase, tokenize and then do set union
    # size of the union should be one off from the size of the original attribute marker
    # to optimize, we don't consider attribute markers of length greater than marker + 1
    # or smaller than marker - 1
    # TODO: lowercase

    marker_split = marker.split(" ")
    for attr, score in attr_vocab.items():
        attr_split = attr.split(" ")
        # if there is a possibility of word edit distance, consider it
        if len(attr_split) >= len(marker_split) - 1 and len(attr_split) <= len(marker_split) + 1:



