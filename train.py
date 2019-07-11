import os
import argparse

import torch
import torch.optim as optim

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

from src.models import DeleteOnly

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.common.util import START_SYMBOL, END_SYMBOL

from src.dataset_readers import DeleteOnlyDatasetReader
from src.predictor import DeleteOnlyPredictor
from src.evaluation import read_test_file, predict_outputs, calculate_bleu

torch.manual_seed(1)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

reader = DeleteOnlyDatasetReader()

train_neg_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.train.neg')
train_pos_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.train.pos')

validate_neg_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.dev.neg')
validate_pos_file = os.path.join(os.path.dirname(__file__), 'data/sentiment.dev.pos')

test_neg2pos_file = os.path.join(os.path.dirname(__file__), 'data/reference.neg.pos')
test_pos2neg_file = os.path.join(os.path.dirname(__file__), 'data/reference.pos.neg')

negative_train_dataset = reader.read(train_neg_file)
positive_train_dataset = reader.read(train_pos_file)

negative_validation_datset = reader.read(validate_neg_file)
positive_validation_datset = reader.read(validate_pos_file)

neg2pos_test = read_test_file(test_neg2pos_file)
pos2neg_test = read_test_file(test_pos2neg_file)

train_data = negative_train_dataset + positive_train_dataset
validation_data = negative_validation_datset + positive_validation_datset

vocab = Vocabulary.from_instances(train_data + validation_data)

EMBEDDING_DIM = 128
HIDDEN_DIM = 512

# TODO: consider using pretrained word embeddings
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
attribute_embedder = Embedding(num_embeddings=2, embedding_dim=EMBEDDING_DIM)
word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = DeleteOnly(word_embedder,
                   attribute_embedder,
                   lstm,
                   vocab,
                   max_decoding_steps=20,
                   beam_size=8,
                   scheduled_sampling_ratio=0.5)

if torch.cuda.is_available():
    cuda_device = 1
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.Adadelta(model.parameters())
iterator = BucketIterator(batch_size=256, sorting_keys=[("content", "num_tokens")])
iterator.index_with(vocab)

# TODO: should probably provide a patience
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_data,
                  validation_dataset=validation_data,
                  patience=10,
                  num_epochs=35,
                  cuda_device=cuda_device)

trainer.train()

checkpoints_path = "./ckpt"
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

print("Saving model")
with open(checkpoints_path + "/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files(checkpoints_path + "/vocabulary")

predictor = DeleteOnlyPredictor(model, reader)
predicted_outputs = predict_outputs(model, neg2pos_test, predictor, "positive")

bleu_score = calculate_bleu(predicted_outputs)
print(bleu_score)


# And here's how to reload the model.
# vocab2 = Vocabulary.from_files("/tmp/vocabulary")
# model2 = DeleteOnly(word_embedder, attribute_embedder, lstm, vocab2)
# with open("/tmp/model.th", 'rb') as f:
#     model2.load_state_dict(torch.load(f))
# if cuda_device > -1:
#     model2.cuda(cuda_device)
