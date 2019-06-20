from overrides import overrides

from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import Token

from allennlp.common.util import JsonDict

from tools.data import get_content

from allennlp.common.util import START_SYMBOL, END_SYMBOL

class DeleteOnlyPredictor(Predictor):
    """
    Predictor for Delete Only model
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str, attribute: str) -> JsonDict:
        assert attribute in ["positive", "negative"]
        return self.predict_json({"sentence": sentence, "attribute": attribute})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        attribute = json_dict["attribute"]
        sentence = sentence.strip().split()

        content = get_content(sentence, attribute)

        # guaranteeing that we don't have empty inputs
        content.insert(0, START_SYMBOL)
        content.append(END_SYMBOL)

        sentence.insert(0, START_SYMBOL)
        sentence.append(END_SYMBOL)
        return self._dataset_reader.text_to_instance([Token(word) for word in content], attribute,
                                                     [Token(word) for word in sentence])
