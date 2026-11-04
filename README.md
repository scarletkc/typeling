# word-predictor

Offline English autocomplete for text input: completes the word being typed and predicts the next word, in pure Python.

## Features

- **Offline and private.** No network access at runtime; what the user types never leaves the machine.
- **Pure Python, no dependencies.** The runtime uses only the standard library (Python 3.13+).
- **Small and fast.** The bundled model is at most 5 MB, loads in milliseconds and answers a keystroke in tens of
  microseconds.
- **Completion and prediction.** Completes the current word (`Let me kn` → `ow`) and predicts the next one after a
  space (`Thank you so ` → `much`).
- **Natural output.** Handles contractions such as *I'm* and *don't* and restores casing (*I*, *English*, sentence
  starts, ALL-CAPS input).
- **Learns locally.** Adapts to the user's own words and phrases and stores them in a local JSON file.
- **Unobtrusive.** Ghost text is shown only above a confidence threshold, and offensive words are never suggested by
  default.
- **Permissively licensed data.** The model is built only from openly licensed corpora (ODC-By, CC BY, Apache-2.0).

## Quick start

```bash
pip install .
```

### Python API

```python
from word_predictor import Predictor

p = Predictor()
p.suggest("Let me kn")         # 'ow'    ghost text to display after the cursor
p.suggest("Thank you so ")     # 'much'  after a space, the next word is predicted
p.suggest("Hello,")            # ''      nothing is shown when no confident suggestion exists
p.topk("I'll see you ", 3)     # [(word, probability), ...] best full words, ignoring the display thresholds

p = Predictor(user="path/to/user.json")   # enable persistent learning
p.learn("Our project is called Zorblaxa.")
p.suggest("I love Zorb")       # 'laxa'  learned words are suggested too
p.save()                       # write the user model back to user.json
```

`suggest(text)` and `topk(text, k)` take the text before the cursor. `Predictor` also accepts `model` (path to an
alternative model file) and the keyword-only options `next_threshold` (default 0.15), `complete_threshold` (0.3),
`cache_weight` (0.1), `allow_blocked` (False) and `use_context` (True).

### Terminal demo

```bash
python -m word_predictor                  # or: word-predictor
python -m word_predictor --no-learn       # do not learn or save your words
python -m word_predictor --bench          # measure per-keystroke latency
echo "I wan" | python -m word_predictor   # pipe mode: prints "I wan[t]"
```

| Key | Action |
|---|---|
| Tab / Right | accept the suggestion |
| Esc | hide the suggestion |
| Ctrl+W | delete the previous word |
| Enter | new line (the finished line is learned unless `--no-learn` is given) |
| Ctrl+C | quit |

Learned words are stored in `~/.word_predictor/user.json` and saved on exit.

## How it works

- **Tokenizer** (`text.py`): words (with inner apostrophes), numbers, sentence boundaries, commas and other
  punctuation. Training and runtime share the same rules.
- **Language model** (`model.py`): an interpolated modified Kneser-Ney trigram model over a 60,000-word vocabulary,
  pruned with the weighted-difference criterion (Seymore & Rosenfeld) to fit a 5 MB budget, with probabilities
  quantized to one byte. Words are stored alphabetically, so all completions of a prefix form one contiguous id
  range. The binary format is documented at the top of `model.py`.
- **Scoring** (`predictor.py`): the n-gram probability is mixed with a document cache (words used earlier in the same
  text) and the user model (`user.py`, unigram and bigram counts of what the user typed), whose weight grows as more
  text is learned.
- **Display**: a next-word prediction is shown only if its probability is at least `next_threshold`; a completion
  only if its probability given the typed prefix is at least `complete_threshold`.

## Evaluation

`bench/eval.py` simulates typing the held-out dev split of each corpus (created by `uv run training/build.py fetch`)
and reports how much the suggestions help. For each word it "types" 0, 1, 2, ... letters; when the top suggestion is
the target word and passes the display threshold, the simulated user accepts it with one keystroke.

```bash
python bench/eval.py                                                # dev split, all sources
python bench/eval.py --limit 2000 --sources soda fineweb --no-save   # quick run
```

| Option | Meaning |
|---|---|
| `--limit N` | at most N target words per source (default 20000, 0 = no limit) |
| `--sources ...` | corpora to evaluate (default: soda wildchat oasst2 fineweb) |
| `--model PATH` | evaluate another model file |
| `--no-context` | ablation: ignore the context (typed letters and word frequency only) |
| `--learn` | call `learn()` after each segment to simulate user learning |
| `--cache W` | override the cache weight |
| `--grid` | also print KSR / precision for a grid of display thresholds |
| `--final` | use the sealed test split (final comparisons only) |
| `--note TEXT` | note stored with the results |
| `--no-save` | do not append the results to `bench/results.jsonl` |

| Metric | Definition |
|---|---|
| KSR | keystroke savings rate: 1 − keystrokes used / letters in the target words |
| Precision | share of displayed suggestions that were accepted |
| Shown | suggestions displayed per word |
| KSR (always shown) | KSR if every top suggestion were displayed (both thresholds 0) |
| acc@i | share of words whose top candidate is correct after typing i letters (acc@0 = next-word accuracy) |
| top3@0 | share of words among the top 3 candidates before any letter is typed |
| OOV | share of words missing from the model vocabulary |

Results are reported per source and pooled into *chat* (SODA, WildChat, oasst2) and *writing* (FineWeb).

## Results

<!-- Filled in after the final model build. -->

## Building the model

The build script is a [uv](https://docs.astral.sh/uv/) script; its dependencies (numpy, pyarrow, huggingface_hub)
are installed automatically into an isolated environment.

```bash
uv run training/build.py fetch    # download and clean the corpora into data/corpus/
uv run training/build.py model    # count, prune and export word_predictor/en.bin (about 30 s)
uv run training/build.py          # both steps
```

`fetch` skips sources that are already downloaded and accepts `--sources`. `model` accepts `--out PATH` (default
`word_predictor/en.bin`), `--budget BYTES` (default 5e6), `--min-count N` (drop n-grams seen fewer than N times,
default 2) and `--workers N` (default 8).

Each corpus is cleaned (emails, URLs and phone numbers are removed; chat sources keep only turns that look typed by a
person), and about 1% is held out as the dev split and 1% as the test split by content hash.

| Corpus | Dataset | License | Used for |
|---|---|---|---|
| FineWeb | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By 1.0 | writing |
| SODA | [allenai/soda](https://huggingface.co/datasets/allenai/soda) | CC BY 4.0 | everyday dialogue (generated by InstructGPT) |
| WildChat-1M | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-By 1.0 | real user-typed turns |
| OpenAssistant oasst2 | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | real user-typed turns |

The [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) English word list
(CC BY 4.0) is used only to mark words that are never suggested.

## Project layout

```text
word_predictor/        runtime package (standard library only)
  predictor.py         public API: Predictor
  model.py             binary model format and n-gram scoring
  text.py              tokenizer shared by training and runtime
  user.py              local user model
  demo.py              terminal demo (python -m word_predictor)
  en.bin               bundled model
training/build.py      corpus download and model build
bench/eval.py          typing-simulation benchmark
tests/                 unit tests (python -m unittest discover tests)
```

## License

The code is licensed under the [Apache License 2.0](LICENSE). The bundled model `word_predictor/en.bin` is licensed
under the [Open Data Commons Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/1-0/); see
[NOTICE](NOTICE) for the required attributions.
