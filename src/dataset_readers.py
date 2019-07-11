import os

from typing import Iterator, List, Dict
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from tools.data import get_content

from allennlp.common.util import START_SYMBOL, END_SYMBOL

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
        attribute = "negative" if "neg" in file else "positive"
        with open(file_path) as f:
            for line in f:
                sentence = line.strip().split()

                content = get_content(sentence, attribute)

                # guaranteeing that we don't have empty inputs
                content.insert(0, START_SYMBOL)
                content.append(END_SYMBOL)

                sentence.insert(0, START_SYMBOL)
                sentence.append(END_SYMBOL)
                yield self.text_to_instance([Token(word) for word in content], attribute,
                                            [Token(word) for word in sentence])
