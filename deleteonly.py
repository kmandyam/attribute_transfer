import os

import torch

from typing import Iterator, List, Dict
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from tools.data import get_content

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

torch.manual_seed(1)

class DeleteOnlyDatasetReader(DatasetReader):
    """
    DatasetReader for attribute transfer data, one sentence per line, like
        Line: The apple juice was amazing
        Content: The apple juice
        Attribute: Positive
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, content: List[Token], attribute: str,
                         target: List[Token] = None) -> Instance:

        content_field = TextField(content, self.token_indexers)
        attribute_field = LabelField(attribute)
        fields = {"content": content_field, "attribute": attribute_field}

        if target:
            target_field = TextField(target, self.token_indexers)
            fields["target"] = target_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        path, file = os.path.split(file_path)
        attribute = "negative" if "0" in file else "positive"
        with open(file_path) as f:
            for line in f:
                sentence = line.strip().split()
                content = get_content(sentence, attribute)
                yield self.text_to_instance([Token(word) for word in content], attribute,
                                            [Token(word) for word in sentence])


reader = DeleteOnlyDatasetReader()

train_neg_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.train.0')
train_pos_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.train.1')

validate_neg_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.dev.0')
validate_pos_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.dev.1')

negative_train_dataset = reader.read(train_neg_file)
positive_train_dataset = reader.read(train_pos_file)

negative_validation_datset = reader.read(validate_neg_file)
positive_validation_datset = reader.read(validate_pos_file)

train_data = negative_train_dataset + positive_train_dataset
validation_data = negative_validation_datset + positive_validation_datset

vocab = Vocabulary.from_instances(train_data + validation_data)

EMBEDDING_DIM = 128
HIDDEN_DIM = 512

# TODO: consider using pretrained word embeddings
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})