from typing import List, Optional

from pygtrie import Trie
import torch
from transformers import StoppingCriteria


class SuffixCriteria(StoppingCriteria):
    '''
    This class can be used to stop the text generation whenever the
    suffix of the generated tokens matches one of the given suffixes.

    Args:
        suffixes_ids: list[list[int]]
            List of sequence of tokens in the vocabulary
    '''

    def __init__(self, suffixes_ids: List[List[int]]):
        self.reversed_suffixes_ids = Trie()
        for suffix_ids in suffixes_ids:
            self.reversed_suffixes_ids[suffix_ids[::-1]] = True

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        reversed_tokens = input_ids[0, :].flip(0).tolist()
        return self.reversed_suffixes_ids.shortest_prefix(reversed_tokens)[0] is not None

