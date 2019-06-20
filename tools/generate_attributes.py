from typing import List, Set, Tuple, Dict
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk import ngrams
from tqdm import tqdm

SALIENCE_THRESHOLD = 15

class SalienceCalculator(object):
    def __init__(self, src_corpus: List[str], tgt_corpus: List[str], tokenize):
        self.vectorizer = CountVectorizer(tokenizer=tokenize)

        src_count_matrix = self.vectorizer.fit_transform(src_corpus)
        self.src_vocab = self.vectorizer.vocabulary_
        self.src_counts = np.sum(src_count_matrix, axis=0)
        self.src_counts = np.squeeze(np.asarray(self.src_counts))

        tgt_count_matrix = self.vectorizer.fit_transform(tgt_corpus)
        self.tgt_vocab = self.vectorizer.vocabulary_
        self.tgt_counts = np.sum(tgt_count_matrix, axis=0)
        self.tgt_counts = np.squeeze(np.asarray(self.tgt_counts))

    def salience(self, feature, attribute='negative', lmbda=0.5):
        assert attribute in ['negative', 'positive']

        # get counts of ngram in src vocab
        if feature not in self.src_vocab:
            src_count = 0.0
        else:
            src_count = self.src_counts[self.src_vocab[feature]]

        # get counts of ngram in target vocab
        if feature not in self.tgt_vocab:
            tgt_count = 0.0
        else:
            tgt_count = self.tgt_counts[self.tgt_vocab[feature]]

        # calculate salience
        if attribute == 'negative':
            return (src_count + lmbda) / (tgt_count + lmbda)
        else:
            return (tgt_count + lmbda) / (src_count + lmbda)

def generate_attribute_vocabulary() -> None:
    negative_file = os.path.join(os.path.dirname(__file__), '../data/sentiment.train.neg')
    positive_file = os.path.join(os.path.dirname(__file__), '../data/sentiment.train.pos')

    vocab_file = os.path.join(os.path.dirname(__file__), '../data/vocab')

    # create a set of all words in the vocab
    vocab = set([w.strip() for i, w in enumerate(open(vocab_file))])

    # create list of list of strings in sentences
    negative_sentences = [l.strip().split() for l in open(negative_file)]
    positive_sentences = [l.strip().split() for l in open(positive_file)]

    # unk both coropora
    negative_corpus = unk_corpus(negative_sentences, vocab)
    positive_corpus = unk_corpus(positive_sentences, vocab)

    sc = SalienceCalculator(negative_corpus, positive_corpus, tokenize)

    output_path = os.path.join(os.path.dirname(__file__), '../attribute_vocabulary/vocab.attribute')
    attribute_vocab_file = open(output_path, "w")

    # produce attribute vocabularies for both corpora
    calculate_attribute_markers(negative_corpus, sc, attribute_vocab_file)
    calculate_attribute_markers(positive_corpus, sc, attribute_vocab_file)

def calculate_attribute_markers(corpus: List[str], sc: SalienceCalculator,
                                attribute_vocab_file) -> None:
    for sentence in tqdm(corpus):
        for i in range(1, 5):
            # generate all ngrams of length i
            i_grams = ngrams(sentence.split(), i)
            joined = [" ".join(gram) for gram in i_grams]
            # for each n gram, calculate salience and keep if above threshold
            for gram in joined:
                negative_salience = sc.salience(gram, attribute='negative')
                positive_salience = sc.salience(gram, attribute='positive')
                if max(negative_salience, positive_salience) > SALIENCE_THRESHOLD:
                    attribute_vocab_file.write(gram + "\t" + str(negative_salience) +
                                               "\t" + str(positive_salience) + "\n")

# a function to tokenize text into ngrams
def tokenize(text: str) -> List[str]:
    text = text.split()
    grams = []
    for i in range(1, 5):
        i_grams = [" ".join(gram) for gram in ngrams(text, i)]
        grams.extend(i_grams)
    return grams

# removing less common words with unking
def unk_corpus(sentences: List[List[str]], vocab: Set) -> List[str]:
    corpus = []
    for line in sentences:
        # unk the sentence according to the vocab
        line = [w if w in vocab else '<unk>' for w in line]
        corpus.append(' '.join(line))
    return corpus

def retrieve_attribute_vocabulary() -> Tuple[Dict[str, float], Dict[str, float]]:
    # the path to the attribute vocabulary
    attr_vocab_path = os.path.join(os.path.dirname(__file__), '../attribute_vocabulary/vocab.attribute')
    exists = os.path.isfile(attr_vocab_path)

    # if the attribute vocab doesn't exist, create it
    if not exists:
        print("Generating new attribute vocabulary")
        generate_attribute_vocabulary()
        assert os.path.isfile(attr_vocab_path)

    print("Loading attribute vocabulary")
    # load attribute vocab from file
    negative_vocab = {}
    positive_vocab = {}

    for line in open(attr_vocab_path):
        split = line.strip().split()
        negative_salience = float(split[-2])
        positive_salience = float(split[-1])
        attr = ' '.join(split[:-2])
        if negative_salience > SALIENCE_THRESHOLD:
            negative_vocab[attr] = negative_salience
        if positive_salience > SALIENCE_THRESHOLD:
            positive_vocab[attr] = positive_salience

    return negative_vocab, positive_vocab
