from typing import Dict, List, Tuple
import torch
from overrides import overrides

import numpy
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.nn.beam_search import BeamSearch

from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, sequence_cross_entropy_with_logits

class DeleteOnlyBeam(Model):
    def __init__(self,
                 word_embedder: TextFieldEmbedder,
                 attribute_embedder: Embedding,
                 content_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,) -> None:
        super().__init__(vocab)

        self.scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self.start_index = self.vocab.get_token_index(START_SYMBOL, 'tokens')
        self.end_index = self.vocab.get_token_index(END_SYMBOL, 'tokens')

        # TODO: not sure if we need this
        self.bleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self.max_decoding_steps = max_decoding_steps
        self.beam_search = BeamSearch(self.end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source and target vocab tokens and attribute.
        self.word_embedder = word_embedder
        self.attribute_embedder = attribute_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self.content_encoder = content_encoder

        num_classes = self.vocab.get_vocab_size('tokens')

        # TODO: not sure if we need this
        self.attention = None

        # Dense embedding of vocab words in the target space.
        embedding_dim = word_embedder.get_output_dim()
        self.target_embedder = Embedding(num_classes, embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self.encoder_output_dim = self.content_encoder.get_output_dim() + embedding_dim
        self.decoder_output_dim = self.encoder_output_dim

        self.decoder_input_dim = embedding_dim

        self.decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        self.output_projection_layer = Linear(self.decoder_output_dim, num_classes)

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # shape: (group_size, num_classes)
        output_projections, state = self.prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def forward(self,  # type: ignore
                content: Dict[str, torch.Tensor],
                attribute: torch.Tensor,
                target: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        state = self.encode(content, attribute)

        if target:
            state = self.init_decoder_state(state)
            # The `forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self.forward_loop(state, target)
        else:
            output_dict = {}

        if not self.training:
            state = self.init_decoder_state(state)
            predictions = self.forward_beam_search(state)
            output_dict.update(predictions)

        return output_dict

    def forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        content_mask = state["source_mask"]

        batch_size = content_mask.size()[0]

        if target:
            # shape: (batch_size, max_target_sequence_length)
            targets = target["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self.max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = content_mask.new_full((batch_size,), fill_value=self.start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self.scheduled_sampling_ratio:
                input_choices = last_predictions
            elif not target:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self.prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = get_text_field_mask(target)
            loss = self.get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self.end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def encode(self,
               content: Dict[str, torch.Tensor],
               attribute: torch.Tensor) -> Dict[str, torch.Tensor]:

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

        return {
                "source_mask": content_mask,
                "encoder_outputs": encoder_output,
        }

    def init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = state["encoder_outputs"]

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self.decoder_output_dim)
        return state

    def forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self.start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self.beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self.target_embedder(last_predictions)

        # shape: (group_size, target_embedding_dim)
        decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self.decoder_cell(
            decoder_input,
            (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self.output_projection_layer(decoder_hidden)

        return output_projections, state

    @staticmethod
    def get_loss(logits: torch.LongTensor,
                 targets: torch.LongTensor,
                 target_mask: torch.LongTensor) -> torch.Tensor:

        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        return all_metrics
