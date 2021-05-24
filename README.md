# 🤗 Transformers Controllers 🎮

Helper classes to constrain the text generation output by 🤗 [Transformers](https://huggingface.co/transformers/).

This module only supports PyTorch at this moment.

## Installation

```
pip install transformers-controllers
```

## Helpers

Helpers are subclasses of 
[LogitsProcessor, LogitsWarper](https://huggingface.co/transformers/_modules/transformers/generation_logits_process.html) or
[StoppingCriteria](https://huggingface.co/transformers/_modules/transformers/generation_stopping_criteria.html).

`GoodPhrasesLogitsProcessor` specifies which words or phrases can be appeared in the generated text.

`SuffixCriteria` stops the generation when the end of the text matches one of the given suffixes.

## Examples

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    StoppingCriteriaList
)
from transformers_controllers import SuffixCriteria, GoodPhrasesLogitsProcessor

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

seed = 256

prompt = 'This morning, when I was walking in the park, I looked up and'
input_ids = tokenizer.encode(prompt, return_tensors='pt')

print('--- The writing prompt ---\n')
print(prompt)

torch.manual_seed(seed)
output = model.sample(
    input_ids,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50
)

print('--- Without any control ---\n')
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Stop the generation if it hits either of these punctuations.
stopping_criteria = StoppingCriteriaList([
    SuffixCriteria([
        tokenizer.encode(suffix) for suffix in ['.', '!', '?', '...']
    ])
])

torch.manual_seed(seed)
output = model.sample(
    input_ids,
    stopping_criteria=stopping_criteria,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50
)

print('\n--- Stop at the end of the first sentence ---\n')
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Use only these words in the generated output:
logits_processor = LogitsProcessorList([
    GoodPhrasesLogitsProcessor([
        tokenizer.encode(phrase) for phrase in [
            '!', ',', '.', ' saw', ' morning', ' bird', ' I', ' a', ' the', ' in',
        ]
    ], num_beams=1)
])

torch.manual_seed(seed)
output = model.sample(
    input_ids,
    logits_processor=logits_processor,
    stopping_criteria=stopping_criteria,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50
)

print('\n--- Finish a sentence with a given list of words ---\n')
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

And this is the output:
```
--- The writing prompt ---

This morning, when I was walking in the park, I looked up and

--- Without any control ---

This morning, when I was walking in the park, I looked up and saw a Rita Skeeter painting. I was wearing my suit that day in a round-autumn shower. The one we had planned for him was a tailored suit with pink

--- Stop at the end of the first sentence ---

This morning, when I was walking in the park, I looked up and saw a Rita Skeeter painting.

--- Finish a sentence with a given list of words ---

This morning, when I was walking in the park, I looked up and saw a bird, I saw the bird.
```

## References

[1] [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)

[2] [Summary of the tokenizers](https://huggingface.co/transformers/master/tokenizer_summary.html)
