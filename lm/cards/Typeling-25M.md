---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: SupraLabs/Supra2-Medium-Base
base_model_relation: finetune
datasets:
- HuggingFaceFW/fineweb
- allenai/soda
- allenai/WildChat-1M
- OpenAssistant/oasst2
- common-pile/ubuntu_irc_filtered
tags:
- typing
- autocomplete
- word-prediction
- next-word-prediction
- keyboard
- typo-correction
---

# Typeling-25M

A small English language model for typing prediction: it completes the word being typed and predicts the next word,
for ghost text or a candidate bar in chat and everyday writing. 25.4M parameters, 50.8 MB at 16 bits (27.6 MB as Q8_0
in [Typeling-25M-GGUF](https://huggingface.co/ScarletKc/Typeling-25M-GGUF)). Through the PyTorch decoder it takes about
25 ms per suggestion on a high-end laptop CPU and about 549 MB of memory; the word-level decoder does not run on the
GGUF file yet.

## Quick start

```bash
pip install "typeling[lm]"
typeling --lm                       # terminal demo: ghost text and a candidate bar
```

```python
from typeling.lm import LMPredictor

p = LMPredictor()                   # downloads ScarletKc/Typeling-25M
p.suggest("Let me kn")              # 'ow'
p.topk("Thank you so ", 3)          # [('much', 0.98), ...]
p.topk("See you tomm", 3)           # [('tomorrow', 0.90), ('tommy', 0.07), ...]   typos are repaired
```

The model predicts tokens; the [Typeling decoder](https://github.com/scarletkc/typeling) turns them into word
suggestions. It searches the words that fit the letters typed so far and gives each a probability, repairs typing
errors (adjacent keys, extra or missing letters, swapped letters), mixes in a small n-gram model that pushes non-words
down the list, and learns the user's own words locally. All results below are with this decoder and the 16-bit weights.

The model is downloaded once and typed text never leaves the machine, but loading it still asks Hugging Face for the
latest revision unless `HF_HUB_OFFLINE=1` is set ([issue #1](https://github.com/scarletkc/typeling/issues/1)).

### With Transformers only

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("ScarletKc/Typeling-25M")
model = AutoModelForCausalLM.from_pretrained("ScarletKc/Typeling-25M")
ids = [tok.eos_token_id] + tok("Thank you so", add_special_tokens=False)["input_ids"]
top = model(torch.tensor([ids])).logits[0, -1].topk(3).indices
print([tok.decode(i) for i in top])  # [' much', ' ', ' very']
```

- Texts in training were separated by `</s>`: start the input with it.
- The tokenizer puts a space before every text, so the first word is `" Thank"`.
- Predictions are best when the text ends right after a word; the decoder moves a trailing space into the
  prediction itself.
- Tested with Transformers 4.57 and 5.22.

## Results

Typing simulation on held-out test splits that were not used for any decision: for every word, the simulated user
types 0, 1, 2, ... letters and accepts the suggestion with one keystroke as soon as it is right.

- **KSR** (keystroke savings rate) with ghost text: the top suggestion is shown when its probability is at least 0.3
  and it extends the typed letters.
- **KSR@3** with a candidate bar: the top 3 candidates are always shown, and picking one counts as one keystroke.
- The first 2,000 words per source (for FineWeb, three documents). Chat is SODA, WildChat and oasst2, casual is Ubuntu
  IRC, writing is FineWeb.

| | n-gram, 5 MB | n-gram, 23 MB | **Typeling-25M** |
|---|---|---|---|
| Chat: KSR / KSR@3 | 31.6% / 48.8% | 34.1% / 50.6% | **43.1% / 58.0%** |
| Casual: KSR / KSR@3 | 22.7% / 38.9% | 24.4% / 40.4% | **29.8% / 46.8%** |
| Writing: KSR / KSR@3 | 22.3% / 40.6% | 23.9% / 41.5% | **32.4% / 49.9%** |
| Chat: precision of ghost text | 51.0% | 51.2% | **54.7%** |

The n-gram models are Typeling's pure-Python kernel (an interpolated Kneser-Ney trigram model built from the same
corpus; 23 MB is the unpruned model). It is not the strongest possible n-gram model (no higher orders, larger
vocabularies or lower count cutoffs were tried), and the language model also has the base model's 20B tokens of
pretraining behind it.

Next word, before any letter is typed: the right word on top / among the top 3.

| Source | n-gram, 23 MB | Typeling-25M |
|---|---|---|
| SODA | 30.0% / 46.9% | 43.4% / 62.5% |
| WildChat | 22.6% / 35.5% | 31.3% / 45.9% |
| oasst2 | 18.9% / 30.7% | 28.1% / 44.5% |
| Ubuntu IRC | 13.9% / 24.0% | 19.0% / 32.4% |
| FineWeb | 15.0% / 27.1% | 24.8% / 37.7% |

Typing errors: the same simulation with every word of 3+ letters mistyped (one error, two in one word out of five;
the first 1,000 words per source, for FineWeb one document). KSR@3 typed correctly -> with errors, and the share of
mistyped words whose intended word is picked from the candidate bar at any step. Corrections appear only in the bar:
ghost text extends what was typed.

| | n-gram, 5 MB | Typeling-25M |
|---|---|---|
| Chat | 49.6% -> 40.5% (71.0%) | 59.4% -> 55.7% (94.5%) |
| Casual | 39.1% -> 29.3% (57.8%) | 46.5% -> 41.3% (88.4%) |
| Writing | 42.3% -> 32.0% (62.9%) | 50.8% -> 45.5% (90.5%) |

The errors come from the same keyboard model that the repair undoes, and earlier words are never mistyped, so these
numbers are optimistic. The share also counts words picked before the error is typed. The n-gram kernel does not
repair typos, and the bar was not measured without repair, so the table compares two systems rather than isolating
repair.

Perplexity per word on the test splits: SODA 17.7, WildChat 180, oasst2 107, Ubuntu IRC 323, FineWeb 175.

Speed on a high-end laptop CPU (AMD Ryzen 9 9955HX, 4 threads, PyTorch, float32), per suggestion over 300 words per
source, mean (p95 / p99): 23–25 ms (39–48 / 51–60 ms) on chat and IRC, 33 ms (63 / 82 ms) on long web text (one
document). Resident memory is 549 MB. Startup is 4.6 s to import PyTorch plus 2.5 s to load the cached model with
`HF_HUB_OFFLINE=1`, or 9.3 to 12.6 s with the default revision check. llama.cpp runs the Q8_0 file at about 12,000
tokens/s for prompts and 2,800 tokens/s for single-token steps on the same 4 threads, but no word-level decoder runs on
it yet, so it has not been through this benchmark.

## Training

- **Base model:** [SupraLabs/Supra2-Medium-Base](https://huggingface.co/SupraLabs/Supra2-Medium-Base), a Qwen3
  architecture (7 layers, hidden size 512, 8 heads, 16,384-token vocabulary, 1,024-token context) pretrained on 20B
  tokens of FineWeb-Edu.
- **Continued pretraining:** all parameters, 150M tokens (about 2.2 epochs of a 67M-token corpus), 4,578 steps of
  32,768 tokens, sequences of 512 tokens. AdamW with learning rate 6e-4, 5% warmup, weight decay 0.1, bf16 mixed
  precision. 58 minutes on one laptop RTX 5070 Ti.
- **Changes for typing:**
  - `</s>` separates texts; its embedding was initialized from the newline token's before training.
  - In 1% of words, `" word"` was re-tokenized as `" "` + `"word"`, so the model also continues text that ends in a
    space.
- **Data:** chat and writing. FineWeb has emails, URLs and phone numbers removed; WildChat, oasst2 and IRC keep only
  turns that look typed by a person and contain none of them, and WildChat also drops turns flagged as toxic or
  redacted; SODA, which is generated, is used as released. About 1% of FineWeb, WildChat and IRC is held out for
  development and 1% for testing; SODA and oasst2 keep their official splits.

| Corpus | License | Words | Share of training |
|---|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By 1.0 | 18.5M | 30% |
| [SODA](https://huggingface.co/datasets/allenai/soda) | CC BY 4.0 | 16.3M | 25% |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (user turns) | ODC-By 1.0 | 7.5M | 27% |
| [Ubuntu IRC](https://huggingface.co/datasets/common-pile/ubuntu_irc_filtered) | Public Domain | 4.4M | 17% |
| [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) (user turns) | Apache-2.0 | 0.4M | 1% |

A 40M-token fine-tuning run on synthetic typos was also tried. Decoded in its own typed mode, it was worse than this
model with typo repair in the decoder on both clean and mistyped text, so this model does not include it.

The [technical report](https://github.com/scarletkc/typeling/releases/latest/download/typeling-technical-report.pdf)
(PDF) describes the model, the decoder and the results;
[docs/model-plan.md](https://github.com/scarletkc/typeling/blob/master/docs/model-plan.md) records the plan and every
experiment.

## Intended use and limitations

- For predicting what a person types in English: completions and next words in chat, messages and everyday writing.
  It is not a chat or instruction model and has little world knowledge.
- English only, ASCII words. Short informal text works best; long technical text less so.
- SODA was generated by InstructGPT, and OpenAI's terms restrict using its outputs to build competing models.
- The base model was trained on FineWeb-Edu, which is filtered from FineWeb, so writing-domain results may be
  slightly optimistic.
- Typing and typo results are simulated, with no user study; the errors do not come from recorded human typing. The
  benchmark cannot show whether learning the user's words helps, because its texts come from many different people.
- Each result uses the first 1,000 to 2,000 words of a split, with no confidence intervals or repeated samples. System
  comparisons are on the same words, but absolute numbers depend on the sample: the 23 MB n-gram scores 28.9% KSR on
  WildChat's development sample and 34.8% on its test sample.
- The model can suggest offensive words or reproduce biases from web and chat text. The Typeling decoder does not
  suggest words from a blocklist, nor common misspellings, but other harmful completions are possible. The model was
  not tested for verbatim reproduction of training text or personal information.
- The candidate bar can still offer non-words right after a finished word (`muchfor`).
- Without the Typeling decoder you get raw next-token predictions, with no word-level probabilities, typo repair or
  filtering.

## License

Apache License 2.0, like the base model. The training data requires attribution: FineWeb (ODC-By 1.0, also subject
to the Common Crawl Terms of Use), SODA (CC BY 4.0, Kim et al. 2022), WildChat-1M (ODC-By 1.0, Zhao et al. 2024),
OpenAssistant oasst2 (Apache-2.0, LAION), and the Ubuntu IRC logs (Public Domain, packaged by the Common Pile v0.1,
Kandpal et al. 2025). See `NOTICE`.
