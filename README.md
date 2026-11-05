# Typeling

Offline English typing prediction: completes the word being typed, predicts the next word and fixes typos.

The `typeling` package has two predictors with the same interface:

- `typeling.Predictor`: a dependency-free n-gram kernel in pure Python with a bundled 5 MB model.
- `typeling.lm.LMPredictor`: the typing decoder for [Typeling-25M](https://huggingface.co/ScarletKc/Typeling-25M),
  a 25M-parameter transformer language model trained for typing, with typo repair (optional extra
  `typeling[lm]`: PyTorch and Transformers).

Training and evaluation of the language model are in `lm/`. The
[technical report](https://github.com/scarletkc/typeling/releases/latest/download/typeling-technical-report.pdf)
(PDF) describes the model, the decoder and the results;
[docs/model-plan.md](https://github.com/scarletkc/typeling/blob/master/docs/model-plan.md) records the plan and every
experiment.

## Features

- **Offline and private.** Predictions run on your machine, and what the user types never leaves it. The n-gram kernel
  needs no network. The language model is downloaded once; loading it still asks Hugging Face for the latest revision
  unless `HF_HUB_OFFLINE=1` is set ([issue #1](https://github.com/scarletkc/typeling/issues/1)).
- **Pure Python, no dependencies.** The n-gram kernel uses only the standard library (Python 3.13+).
- **Small and fast.** The bundled n-gram model is at most 5 MB, loads in milliseconds and answers a keystroke in tens
  of microseconds; the language model answers in tens of milliseconds on a laptop CPU.
- **Completion and prediction.** Completes the current word (`Let me kn` → `ow`) and predicts the next one after a
  space (`Thank you so ` → `much`), as ghost text or as a bar of candidates.
- **Typo repair.** With the language model, mistyped words are corrected from keyboard errors (`See you tomm` →
  *tomorrow*, `What do you thnik` → *think*).
- **Natural output.** Handles contractions such as *I'm* and *don't* and restores casing (*I*, *English*, sentence
  starts, ALL-CAPS input).
- **Learns locally.** Adapts to the user's own words and phrases and stores them in a local JSON file.
- **Unobtrusive.** Ghost text is shown only above a confidence threshold, offensive words are not suggested by
  default, and common misspellings found in real typing (*recieve*, *definately*) are never suggested.
- **Permissively licensed data.** The model is built only from openly licensed corpora (ODC-By, CC BY, Apache-2.0).

## Quick start

```bash
pip install typeling            # n-gram kernel, no dependencies
pip install "typeling[lm]"      # plus the language model decoder (PyTorch, Transformers)
```

### Python API

```python
from typeling import Predictor

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
alternative model file) and the keyword-only options `next_threshold` (default 0.3), `complete_threshold` (0.3),
`cache_weight` (0.1), `allow_blocked` (False) and `use_context` (True).

With the language model:

```python
from typeling.lm import LMPredictor

p = LMPredictor()                    # downloads ScarletKc/Typeling-25M once; or a repo id / local directory
p.topk("See you tomm", 3)            # [('tomorrow', ...), ...]  corrections are candidates too
p.suggest("Let me kn")               # 'ow'    ghost text only extends what was typed
p.topk("I'm going to ", 5)           # the search is tuned for the top 3, further candidates are best effort
```

`LMPredictor(model, user)` takes the same user model file as `Predictor`, with the same `learn()` and `save()`. It
also accepts `device` (`"cpu"` or `"cuda"`, default: CUDA if available), `typo_weight` (default 0.003, 0 turns typo
repair off), `ngram_weight` (0.1, share of the n-gram model in the probabilities), `context_chars` (1000),
`beam` (8), `allow_blocked` and the display thresholds.

### Terminal demo

```bash
python -m typeling                  # or: typeling
python -m typeling --lm             # use Typeling-25M (needs typeling[lm]); --lm DIR for a local model
python -m typeling --no-learn       # do not learn or save your words
python -m typeling --bench          # measure per-keystroke latency
echo "I wan" | python -m typeling   # pipe mode: prints "I wan[t]"
```

The line shows the best suggestion as grey ghost text; the bar below it shows the top candidates.

| Key | Action |
|---|---|
| Tab / Right | accept the highlighted candidate (the first one by default; it replaces a mistyped word) |
| Up / Down | move the highlight in the candidate bar |
| Esc | hide the suggestions |
| Ctrl+W | delete the previous word |
| Enter | new line (the finished line is learned unless `--no-learn` is given) |
| Ctrl+C | quit |

Learned words are stored in `~/.typeling/user.json` (used by both models) and saved on exit.

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
- **Spelling filter**: the chat corpora contain real typos, so at build time every vocabulary word that is missing
  from a speller dictionary (and from a short list of internet slang such as *lol*, *idk*, *tbh*), is usually written
  in lowercase, and is one edit (two for words of 8+ letters) away from a dictionary word at least 20 times as common
  is marked as a misspelling. Such words still serve as context but are never suggested.

## Evaluation

`bench/eval.py` simulates typing the held-out dev split of each corpus (created by `uv run training/build.py fetch`)
and reports how much the suggestions help. For each word it "types" 0, 1, 2, ... letters; when the top suggestion is
the target word, extends the typed letters and passes the display threshold, the simulated user accepts it with one
keystroke (ghost text);
with a candidate bar, the user picks the target with one keystroke as soon as it is among the top 3. `lm/eval.py`
runs the same simulation on the language model, and `--typos` repeats it with every word mistyped.

```bash
python bench/eval.py                                                # dev split, all sources
python bench/eval.py --limit 2000 --sources soda fineweb --no-save   # quick run
```

| Option | Meaning |
|---|---|
| `--limit N` | at most N target words per source (default 20000, 0 = no limit) |
| `--sources ...` | corpora to evaluate (default: soda wildchat oasst2 irc fineweb) |
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
| KSR | keystroke savings rate with ghost text: 1 − keystrokes used / letters in the target words |
| Precision | share of displayed suggestions that were accepted |
| Shown | suggestions displayed per word |
| KSR@3 | keystroke savings rate with a bar of the top 3 candidates, always shown |
| acc@i | share of words whose top candidate is correct after typing i letters (acc@0 = next-word accuracy) |
| top3@i | share of words among the top 3 candidates after typing i letters |
| OOV | share of words missing from the model vocabulary |

Results are reported per source and pooled into *chat* (SODA, WildChat, oasst2), *casual* (Ubuntu IRC) and
*writing* (FineWeb).

## Results

**Typeling-25M against the n-gram kernel**, on the sealed test split (2,000 words per source; KSR with ghost text /
KSR@3 with a bar of 3 candidates):

| Domain | n-gram, bundled 5 MB | n-gram, unpruned 23 MB | Typeling-25M |
|---|---|---|---|
| chat (SODA, WildChat, oasst2) | 31.6% / 48.8% | 34.1% / 50.6% | **43.1% / 58.0%** |
| casual (Ubuntu IRC) | 22.7% / 38.9% | 24.4% / 40.4% | **29.8% / 46.8%** |
| writing (FineWeb) | 22.3% / 40.6% | 23.9% / 41.5% | **32.4% / 49.9%** |

With every word of 3+ letters mistyped, chat KSR@3 drops from 49.6% to 40.5% with the n-gram kernel and from 59.4%
to 55.7% with Typeling-25M, whose candidates include corrections: the intended word is picked from the bar for 94.5%
of the mistyped words, against 71.0% (the n-gram kernel does not repair typos). Ghost text only extends what was
typed, so it cannot correct. Next-word
accuracy, typo results per domain, latency and training details are in the
[model card](https://huggingface.co/ScarletKc/Typeling-25M) and
[docs/model-plan.md](https://github.com/scarletkc/typeling/blob/master/docs/model-plan.md).

**The n-gram kernel** on the dev split, bundled 5.00 MB model, default thresholds (next word 0.3, completion 0.3),
20,000 target words per source (oasst2: all 9,251), measured with `python bench/eval.py`:

| Domain | Source | KSR | Precision | Shown | KSR@3 | acc@0 | top3@0 | top3@1 | top3@2 | OOV |
|---|---|---|---|---|---|---|---|---|---|---|
| chat | SODA | 37.3% | 58.3% | 1.44 | 54.5% | 28.6% | 46.6% | 80.8% | 89.6% | 0.3% |
| chat | WildChat | 30.3% | 47.8% | 1.59 | 46.7% | 21.3% | 32.4% | 59.4% | 69.5% | 2.0% |
| chat | oasst2 | 26.2% | 46.4% | 1.62 | 44.3% | 18.1% | 30.3% | 61.3% | 72.4% | 1.3% |
| casual | Ubuntu IRC | 22.8% | 43.8% | 1.58 | 39.6% | 13.8% | 24.5% | 54.9% | 69.6% | 1.7% |
| writing | FineWeb | 23.1% | 43.2% | 1.68 | 41.7% | 14.3% | 23.8% | 51.1% | 64.4% | 2.9% |
| **chat** | pooled | **32.1%** | **51.5%** | 1.54 | **49.1%** | | | | | |
| **casual** | pooled | **22.8%** | **43.8%** | 1.58 | **39.6%** | | | | | |
| **writing** | pooled | **23.1%** | **43.2%** | 1.68 | **41.7%** | | | | | |

- **Context matters.** With `--no-context` (typed letters and word frequency only), KSR with every suggestion shown
  drops to 15.3% (chat), 13.8% (casual) and 14.4% (writing), and next-word accuracy to about 2%.
- **Thresholds trade savings for precision.** Lowering both thresholds to 0.15 raises chat KSR to 34.9% at 41%
  precision; raising them to 0.5 gives 27.9% at 65%. `python bench/eval.py --grid` prints the full grid.
- **Speed.** A query takes 41–115 µs on average (p99 under 0.7 ms on long FineWeb documents) in pure Python on a
  laptop CPU; the model loads in under 50 ms. Typeling-25M takes 23–25 ms on chat and 33 ms on long text (PyTorch,
  4 CPU threads).

## Building the model

The build script is a [uv](https://docs.astral.sh/uv/) script; its dependencies (numpy, pyarrow, huggingface_hub)
are installed automatically into an isolated environment.

```bash
uv run training/build.py fetch    # download and clean the corpora into data/corpus/
uv run training/build.py model    # count, prune and export typeling/en.bin (about 30 s)
uv run training/build.py          # both steps
```

`fetch` skips sources that are already downloaded and accepts `--sources`. `model` accepts `--out PATH` (default
`typeling/en.bin`), `--budget BYTES` (default 5e6), `--min-count N` (drop n-grams seen fewer than N times,
default 2) and `--workers N` (default 8).

FineWeb has emails, URLs and phone numbers removed; WildChat, oasst2 and IRC keep only turns that look typed by a person
and contain none of them; SODA is used as released. About 1% of FineWeb, WildChat and IRC is held out as the dev split
and 1% as the test split, by a hash of the document ID (FineWeb) or of the text (WildChat, IRC); SODA and oasst2 keep
their official splits.

| Corpus | Dataset | License | Used for |
|---|---|---|---|
| FineWeb | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By 1.0 | writing |
| SODA | [allenai/soda](https://huggingface.co/datasets/allenai/soda) | CC BY 4.0 | everyday dialogue (generated by InstructGPT) |
| WildChat-1M | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-By 1.0 | real user-typed turns (first 7.5M training words) |
| OpenAssistant oasst2 | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | real user-typed turns |
| Ubuntu IRC | [common-pile/ubuntu_irc_filtered](https://huggingface.co/datasets/common-pile/ubuntu_irc_filtered) | Public Domain | informal internet chat (messages that look like commands or code are dropped) |

The bundled model was built from 55M tokens: FineWeb 18.5M words, SODA 16.3M, WildChat 7.5M, Ubuntu IRC 4.4M and
oasst2 0.4M.

The [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) English word list
(CC BY 4.0) is used only to mark offensive words, and the American, British and Canadian word lists of the
[English Speller Database](https://github.com/en-wl/wordlist) (ESDB, formerly SCOWL; MIT-like license) only to
detect misspellings.

## Project layout

```text
typeling/              runtime package (standard library only, except lm.py)
  predictor.py         public API: Predictor
  model.py             binary model format and n-gram scoring
  text.py              tokenizer shared by training and runtime
  user.py              local user model
  keyboard.py          QWERTY typing-error model
  lm.py                language model decoder: LMPredictor (typeling[lm])
  demo.py              terminal demo (python -m typeling)
  en.bin               bundled model
training/build.py      corpus download and n-gram model build
bench/eval.py          typing-simulation benchmark
tests/                 unit tests (python -m unittest discover tests)
lm/                    Typeling-25M: data, training, typo experiments and benchmark (PyTorch, uv scripts)
docs/model-plan.md     plan and results for Typeling-25M
app/windows/           prototype Windows input method: TSF text service (Rust) and engine process (see its README)
```

## License

The code is licensed under the [Apache License 2.0](https://github.com/scarletkc/typeling/blob/master/LICENSE). The
bundled model `typeling/en.bin` is licensed under the
[Open Data Commons Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/1-0/), and Typeling-25M
under the Apache License 2.0; see [NOTICE](https://github.com/scarletkc/typeling/blob/master/NOTICE) for the required
attributions.
