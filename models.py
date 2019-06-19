from allennlp.models import Model
from typing import Dict

import torch
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, sequence_cross_entropy_with_logits

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding

from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common.util import START_SYMBOL, END_SYMBOL

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
        self.start_index = self.vocab.get_token_index(START_SYMBOL, 'tokens')
        self.end_index = self.vocab.get_token_index(END_SYMBOL, 'tokens')

        self.max_decoding_steps = 20

        self.target_embedder = Embedding(self.num_classes, word_embedder.get_output_dim())

        self.decoder_input_dim = word_embedder.get_output_dim()
        self.decoder_output_dim = content_encoder.get_output_dim() + word_embedder.get_output_dim()

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
        batch_size = content['tokens'].size()[0]

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
        encoder_output = torch.cat((final_encoder_output, attr_embedding), 1)

        if target:
            targets = target['tokens']
            target_sequence_length = targets.size()[1]
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self.max_decoding_steps

        # produces tensor: (batch size x hidden dim + embedding dim)
        decoder_hidden = encoder_output

        # produces tensor: (batch size x hidden dim)
        decoder_context = content_encoding.new_zeros(batch_size, self.decoder_output_dim)

        last_predictions = None
        step_logits = []
        step_probabilities = []
        step_predictions = []

        # TODO: a problem: the mask doesn't match the encoder output

        for timestep in range(num_decoding_steps):
            if timestep == 0:
                # For the first timestep, when we do not have targets, we input start symbols.
                input_choices = content_mask.new_full((batch_size,), fill_value=self.start_index)
            else:
                input_choices = last_predictions

            # produces tensor: (batch size x embedding dim)
            decoder_input = self.target_embedder(input_choices)

            decoder_hidden, decoder_context = self.decoder_cell(decoder_input, (decoder_hidden, decoder_context))

            output_projections = self.output_projection_layer(decoder_hidden)

            step_logits.append(output_projections.unsqueeze(1))

            class_probabilities = F.softmax(output_projections, dim=-1)

            _, predicted_classes = torch.max(class_probabilities, 1)

            step_probabilities.append(class_probabilities.unsqueeze(1))

            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        logits = torch.cat(step_logits, 1)

        class_probabilities = torch.cat(step_probabilities, 1)

        all_predictions = torch.cat(step_predictions, 1)

        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if target:
            target_mask = get_text_field_mask(target)
            loss = self.get_loss(logits, target['tokens'], target_mask)
            output_dict["loss"] = loss

        return output_dict

    def get_loss(self,
                 logits: torch.LongTensor,
                 targets: torch.LongTensor,
                 target_mask: torch.LongTensor) -> torch.LongTensor:

        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)

        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)

        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

        return loss

    # TODO: override the decode function
