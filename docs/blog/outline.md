# Blog post outline: X Article

Images are in `docs/blog/images/` (regenerate with `uv run docs/blog/make_images.py` and
`uv run docs/blog/make_demo.py`). Every number below is from the technical report (v0.1.2) and `docs/model-plan.md`.

X Article editor mapping: the title field takes the H1; use "Heading" for sections and "Subheading" below them; insert
tables, code blocks, LaTeX and dividers from the "+" menu (pasted Markdown is not converted); upload images and the GIF
as media. Cover: `cover.png` (5:2).

## Title options

- Typeling: Teaching a 25M-Parameter Model to Autocomplete, Offline
- I Trained a Keyboard Model in an Hour on a Laptop. Here's What Worked and What Didn't
- Beating n-grams at Autocomplete with a 25M-Parameter Model

## 0. TL;DR (first screen)

- Image: `demo.gif` (real output of Typeling-25M: next word, completion, a typo fixed from the candidate bar).
- Three numbers: 43.1% of keystrokes saved on held-out chat (an unpruned n-gram model: 34.1%); 25M parameters,
  27.6 MB as GGUF; 58 minutes of training on one laptop GPU.
- `pip install typeling`; links: GitHub, Hugging Face (model and GGUF), technical report (PDF). Apache-2.0.

## 1. Why I built this

- Phone and desktop keyboards predict well, but their models are closed; open small language models are trained for
  chat and benchmarks, not for typing, and not for a keyboard's budget.
- Goals: offline and private, permissively licensed data and weights, a kernel other projects can embed.

## 2. What "good autocomplete" means

- Two interfaces: ghost text (one suggestion, Tab to accept) and a bar of 3 candidates.
- Image: `ksr.png` (a real suggestion: "See you to" + ghost "morrow" = 3 keystrokes for 8 letters).
- LaTeX: `\mathrm{KSR} = 1 - \frac{\text{keystrokes used}}{\text{letters in the target words}}`
- The benchmark simulates typing held-out text letter by letter; three domains: chat (SODA, WildChat, oasst2),
  casual (Ubuntu IRC), writing (FineWeb). Development split for every decision, test split once at the end.

## 3. Starting point: an n-gram model that stops improving

- Pure-Python Kneser-Ney trigram kernel, a bundled 5 MB model, tens of microseconds per keystroke.
- Removing pruning (5 MB -> 23 MB) adds only 1.8-2.2 points of keystroke savings: a trigram sees two words of
  context, and more size only adds rarer word combinations.

## 4. Picking a tiny base model

- Requirements: under about 50M parameters, Apache-2.0, an architecture llama.cpp runs.
- Table (dev split, zero-shot KSR %, 2,000 words per source):

| Model | Params | Chat | Casual | Writing |
|---|---|---|---|---|
| n-gram | 5 MB | 29.7 | 22.8 | 23.5 |
| n-gram, unpruned | 23 MB | 31.9 | 24.8 | 25.5 |
| Pythia-14M | 14.1M | 28.5 | 21.8 | 33.6 |
| Pythia-31M | 30.5M | 31.2 | 22.9 | 36.6 |
| Supra2-Medium-Base | 25.4M | 33.5 | 22.8 | 38.6 |
| Supra-50M-Base, 4-bit (simulated) | 51.8M | 35.0 | 23.4 | 41.0 |
| Supra-50M-Base | 51.8M | 35.6 | 24.1 | 42.1 |

- The honest trade-off: the 50M model is 1.3-3.5 points better zero-shot; I took the 25M model for half the
  parameters, although a 4-bit 50M file (about 30 MB, estimated) would be about as large as the 8-bit 25M one.

## 5. One hour of continued pretraining

- 150M tokens (about 2.2 epochs of a 67M-token corpus), all parameters, 58 minutes on a laptop RTX 5070 Ti.
- Five permissively licensed corpora (FineWeb, SODA, WildChat users, Ubuntu IRC, oasst2 users) and their mix.
- Two small tricks: initialize the untrained `</s>` separator from the newline token; re-tokenize 1% of `" word"`
  as `" "` + `"word"` so the raw model handles text ending in a space.
- Image: `gain.png`. Key point (development split): on chat, 7.7 of the 9.3 points over the n-gram model come from
  this hour of training; zero-shot, the base model was only 1.6 points ahead (and 2.0 behind on IRC).

## 6. Turning tokens into words (the technical core)

- The problem: models predict subword tokens, users want words, and the space belongs to the next token.
- Image: `trailing-space.png` ("Let me know " -> raw model: "!" 37%; decoder: if 73%).
- Image: `decoder.png`. Constrained beam search: a prefix phase that only allows tokens agreeing with the typed letters,
  a word phase that adds letter tokens until the model ends the word.
- LaTeX: `P(w \mid c, \tau) = \frac{1}{Z} \sum_{y:\, \mathrm{str}(y) = \ell w} P(y \mid c)\, e(y)`
- The search runs until no hypothesis in the beam can beat the third candidate (approximate, not exact).
- Extras: n-gram fusion (weight 0.1) pushes down non-words like "muchfor"; offensive words and learned misspellings
  ("recieve") are never suggested; a local user model learns your words (a made-up name wins after three uses).

## 7. Typos: fix them in the decoder, not the model

- Noisy-channel repair: candidate corrections for one keyboard error, scored by the language model in the same beam
  ("tomm" -> tomorrow, "thnik" -> think). Image: `demo-typo.png`.
- Corrections appear only in the candidate bar; ghost text only extends what you typed.
- Fine-tuning on synthetic typos (40M tokens, one recipe, decoded in its own mode) was worse on clean and mistyped
  text: 6-10 points lower KSR on clean text. Released without it.
- Table (test split, every word of 3+ letters mistyped; KSR@3 clean -> with typos, Correct@3 in parentheses):

| Domain | n-gram, 5 MB | Typeling-25M |
|---|---|---|
| Chat | 49.6 -> 40.5 (71.0%) | 59.4 -> 55.7 (94.5%) |
| Casual | 39.1 -> 29.3 (57.8%) | 46.5 -> 41.3 (88.4%) |
| Writing | 42.3 -> 32.0 (62.9%) | 50.8 -> 45.5 (90.5%) |

- Caveats to state: two different systems (the n-gram kernel has no typo repair), the bar was not measured without
  repair, Correct@3 also counts words picked before the error, and the simulated errors come from the same keyboard
  model the repair inverts.

## 8. Results

- Image: `results.png` (test split; ghost text and a bar of 3).
- Table (test split, first 2,000 words per source; KSR / KSR@3 %):

| System | Chat | Casual | Writing | Precision (chat) |
|---|---|---|---|---|
| n-gram, 5 MB | 31.6 / 48.8 | 22.7 / 38.9 | 22.3 / 40.6 | 51.0 |
| n-gram, unpruned, 23 MB | 34.1 / 50.6 | 24.4 / 40.4 | 23.9 / 41.5 | 51.2 |
| Typeling-25M | 43.1 / 58.0 | 29.8 / 46.8 | 32.4 / 49.9 | 54.7 |

- Next word before any letter (acc@0, %): SODA 30.0 -> 43.4, WildChat 22.6 -> 31.3, oasst2 18.9 -> 28.1,
  IRC 13.9 -> 19.0, FineWeb 15.0 -> 24.8 (23 MB n-gram -> Typeling).
- Q8_0 GGUF: 27.6 MB, perplexity +0.08% against 16-bit, same top token 97.9% of the time.
- Table (footprint; the real remaining cost):

| | n-gram kernel | Typeling-25M (PyTorch) |
|---|---|---|
| Dependencies | none (pure Python) | PyTorch, Transformers |
| Model file | 5 MB | 50.8 MB (16-bit), 27.6 MB (Q8_0 GGUF) |
| Memory | 54 MB | 549 MB |
| Load | 0.03 s | 2.5 s (+4.6 s to import PyTorch) |
| Time per suggestion | 41-115 µs | 23-25 ms mean on chat, 33 ms on long text (CPU, 4 threads) |

## 9. Things I got wrong along the way (the most readable part)

- Typo fine-tuning looked promising in isolation (after a whole mistyped word, the intended word came first for 84%
  of SODA words) and still lost overall.
- "The top 3 are exact": not true; pruned beam paths never come back.
- An example claimed fusion found "encyclopedia"; at weight 0.1 that was mathematically impossible.
- My benchmark counted corrections as accepted ghost text, which the real `suggest()` never shows; fixed in 0.1.1.
- WildChat's development split is 24% smaller than its test split, and I still don't know why.
- The bar can show non-words after a finished word ("muchfor", "tomorrownight"); they appear briefly in the GIF.
- A reviewer pass over the report caught most of these: worth doing for any project write-up.

## 10. Limitations

- Simulated users and simulated typos; no user study. English only.
- No comparison with larger general models (e.g. SmolLM2-360M, Qwen2.5-0.5B).
- No confidence intervals; writing results come from a few documents.
- PyTorch only: no word-level decoder on the GGUF file yet.

## 11. Try it, reuse it, what's next

- Code block:

```bash
pip install typeling            # n-gram kernel, no dependencies
pip install "typeling[lm]"      # plus the language model decoder
typeling --lm                   # terminal demo
```

- Code block:

```python
from typeling.lm import LMPredictor

p = LMPredictor()                  # downloads ScarletKc/Typeling-25M once
p.suggest("Let me kn")             # 'ow'
p.topk("See you tomm", 3)          # [('tomorrow', 0.90), ('tommy', 0.07), ...]
```

- Next: a llama.cpp port of the decoder (issue #2), skipping the Hub check once the model is cached (issue #1), and a
  real input-method integration. Contributions welcome.
- Links: GitHub, Hugging Face (Typeling-25M, Typeling-25M-GGUF), PyPI, technical report (latest release).

## Teaser post (to share the article)

- One or two sentences plus `demo.gif`: a 25M-parameter model, trained for an hour on a laptop, saves 43% of
  keystrokes on chat (an n-gram model: 34%), runs offline, Apache-2.0; link to the article.
