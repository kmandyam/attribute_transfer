from allennlp.models import Model
from typing import Dict

import torch

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding

from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

class DeleteOnly(Model):
    def __init__(self,
                 word_embedder: TextFieldEmbedder,
                 attribute_embedder: Embedding,
                 content_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        self.word_embedder = word_embedder
        self.content_encoder = content_encoder
        self.attribute_embedder = attribute_embedder

        self.num_classes = vocab.get_vocab_size('tokens')

        # initializing decoder variables
        self.max_decoding_steps = 20
        self.decoder_input_dim = content_encoder.get_output_dim() + word_embedder.get_output_dim()
        self.decoder_output_dim = content_encoder.get_output_dim()
        self.decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)
        self.output_projection_layer = Linear(self.decoder_output_dim, self.num_classes)

    def forward(self,
                content: Dict[str, torch.Tensor],
                attribute: torch.Tensor,
                target: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """

        :param content: one key, tokens, with value tensor shape: (batch size x sequence length)
        :param attribute: tensor of shape: (batch size)
        :param target: one key, tokens, with value tensor shape: (batch size x target sequence length)
        :return:
        """
        import pdb; pdb.set_trace()
        # produces tensor: (batch size x sequence length)
        content_mask = get_text_field_mask(content)

        # produces tensor: (batch size x sequence length x embedding dim)
        content_embedding = self.word_embedder(content)

        # produces tensor: (batch size x embedding dim)
        attr_embedding = self.attribute_embedder(attribute)

        # produces tensor: (batch size x sequence length x hidden dim)
        content_encoding = self.content_encoder(content_embedding, content_mask)

        # produces tensor: (batch size x hidden dim)
        final_encoder_output = get_final_encoder_states(
            content_encoding,
            content_mask,
            self.content_encoder.is_bidirectional()
        )

        # produces tensor: (batch size x hidden dim + embedding dim)
        decoder_input = torch.cat((final_encoder_output, attr_embedding), 1)

        return decoder_input