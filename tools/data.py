from typing import List
# TODO: a function which generates a positive and negative attribute vocabulary (plus scores)
# TODO: a function which reads in vocabularies from file if provided
# TODO: a function which, when given a sentence and attribute, returns the content

def get_content(sentence: List[str], attribute: str):
    """
    Splits a sentence into its content (and attribute markers)
    :param sentence: A list of strings representing words in the original sentence
    :param attribute: A string representing the attribute
    :return: A list of strings representing the content
    """
    assert attribute in ["positive", "negative"]

    