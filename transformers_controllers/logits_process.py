from typing import List, Optional

from pygtrie import Trie
import torch
from transformers import PrefixConstrainedLogitsProcessor, LogitsProcessor

class GoodPhrasesLogitsProcessor(PrefixConstrainedLogitsProcessor):
    '''
    This class can be used to constrain which words and phrases
    are allowed in the generated text.

    Args:
        phrases_ids: list[list[int]]
            List of sequence of tokens in the vocabulary
    '''

    def __init__(self, phrases_ids: List[List[int]], num_beams: int):
        allowed_tokens = Trie()
        for phrase_ids in phrases_ids:
            # Given a phrase ABCD, we will generate:
            #        [] -> A
            #       [A] -> B
            #    [B, A] -> C
            # [C, B, A] -> D
            for i in range(len(phrase_ids)):
                reversed_prefix, token = phrase_ids[:i][::-1], phrase_ids[i]
                try:
                    allowed_tokens[reversed_prefix].add(token)
                except KeyError:
                    allowed_tokens[reversed_prefix] = {token}
        for k, v in allowed_tokens.items():
            allowed_tokens[k] = list(v)

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            reversed_tokens = input_ids.flip(0).tolist()
            return allowed_tokens.longest_prefix(reversed_tokens)[1]

        super().__init__(prefix_allowed_tokens_fn, num_beams)

