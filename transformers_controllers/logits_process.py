from typing import List, Optional

from pygtrie import Trie
import torch
from transformers import PrefixConstrainedLogitsProcessor, LogitsWarper


class GoodPhrasesLogitsProcessor(PrefixConstrainedLogitsProcessor):
    '''
    This class can be used to constrain which words and phrases
    are allowed in the generated text.

    Args:
        phrases_ids: list[list[int]]
            List of sequence of tokens in the vocabulary
        num_beams (optional, default = 1): int
            Number of beams in beam search
    '''

    def __init__(self, phrases_ids: List[List[int]], num_beams: int = 1):
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


class ConstantLogitsWarper(LogitsWarper):
    '''
    This class can be used to change the `scores` of the candidate
    tokens to `scores + deltas`, where values in `deltas` can
    be positive to promote a token, or negative for demotion.

    Args:
        deltas: torch.Tensor
            Adjustment of scores for each token in the vocabulary
        num_beams (optional, default = 1): int
            Number of beams in beam search
    '''

    def __init__(self, deltas: torch.Tensor, num_beams: int = 1):
        self.deltas = deltas.squeeze().repeat(num_beams).reshape((num_beams, -1))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores + self.deltas
