import os

import torch
from allennlp.data.vocabulary import Vocabulary

from src.dataset_readers import DeleteRetrieveDatasetReader

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

torch.manual_seed(1)

# load all the data files
train_neg_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.train.neg')
train_pos_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.train.pos')

validate_neg_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.dev.neg')
validate_pos_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.dev.pos')

test_neg2pos_file = os.path.join(os.path.dirname(__file__), 'data/reference.neg.pos')
test_pos2neg_file = os.path.join(os.path.dirname(__file__), 'data/reference.pos.neg')

reader = DeleteRetrieveDatasetReader()

negative_train_dataset = reader.read(train_neg_file)
positive_train_dataset = reader.read(train_pos_file)

negative_validation_datset = reader.read(validate_neg_file)
positive_validation_datset = reader.read(validate_pos_file)

# TODO: at testing time, our attribute markers are gonna be different, write a script that performs retrieve

train_data = negative_train_dataset + positive_train_dataset
validation_data = negative_validation_datset + positive_validation_datset

# read vocabulary
vocab = Vocabulary.from_instances(train_data + validation_data)

# hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 512

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

# TODO: write the Delete Retrieve Model, we have everything to train it


