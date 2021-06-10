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

`ConstantLogitsWarper` makes some words more or less likely to appear in the generated text.

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
from transformers_controllers import (
    SuffixCriteria,
    GoodPhrasesLogitsProcessor,
    ConstantLogitsWarper,
)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

def generate(
    prompt,
    stopping_criteria=None,
    logits_processor=None,
    logits_warper=None,
    max_length=30,
    seed=256
):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    torch.manual_seed(seed)
    with torch.no_grad():
        output = model.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            max_length=max_length
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)

prompt = 'This morning, when I was walking in the park, I looked up and'

print('--- The writing prompt ---\n')
print(prompt)

print('\n--- Without any control ---\n')
generate(prompt)

# Stop the generation if it hits either of these punctuations.
stopping_criteria = StoppingCriteriaList([
    SuffixCriteria([
        tokenizer.encode(suffix) for suffix in ['.', '!', '?', '...']
    ])
])

print('\n--- Stop at the end of the first sentence ---\n')
generate(prompt, stopping_criteria)

# Only use these words in the generated output.
logits_processor = LogitsProcessorList([
    GoodPhrasesLogitsProcessor([
        tokenizer.encode(phrase) for phrase in [
            '!', ',', '.', ' saw', ' morning', ' bird',
            ' lion', ' I', ' a', ' the', ' in', 'me'
        ]
    ])
])

print('\n--- Restrict the choice of words ---\n')
generate(prompt, stopping_criteria, logits_processor)

# Give more weights to some words.
deltas = torch.zeros(tokenizer.vocab_size)
deltas[tokenizer.encode(' lion')] = 2.5
deltas[tokenizer.encode('.')] = 2.5

logits_warper = LogitsProcessorList([
    ConstantLogitsWarper(deltas)
])

print('\n--- Prefer lions than birds ---\n')
generate(prompt, stopping_criteria, logits_processor, logits_warper)
```

And this is the output:
```
--- The writing prompt ---

This morning, when I was walking in the park, I looked up and

--- Without any control ---

This morning, when I was walking in the park, I looked up and
saw a Rita Skeeter painting. I was wearing my suit that day in

--- Stop at the end of the first sentence ---

This morning, when I was walking in the park, I looked up and
saw a Rita Skeeter painting.

--- Restrict the choice of words ---

This morning, when I was walking in the park, I looked up and
saw a bird, I saw the bird.

--- Prefer lions than birds ---

This morning, when I was walking in the park, I looked up and
saw a lion.
```

## References

[1] [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)

[2] [Summary of the tokenizers](https://huggingface.co/transformers/master/tokenizer_summary.html)
