# Plan: Typeling-25M, a small language model for typing prediction

Status: milestones 0–3 done. The base model is chosen (section 12); 150M tokens of continued pretraining (section 13);
typos are handled in the decoder, not by fine-tuning (section 14); the decoder is packaged as `typeling.lm` (optional
extra `typeling[lm]`) with typo repair, a candidate bar, n-gram fusion and the user model; the release files are
exported and the final test-split evaluation meets the v1 success criteria (section 15). Next: the release
(milestone 4), on the owner's go-ahead.

## 1. Goal

Publish Typeling-25M, a small English causal language model that predicts what a person is about to type: the rest
of the current word and the next word. It must be usable on its own, without this repository's autocomplete kernel:
anyone should be able to load it with `transformers` or llama.cpp and run it on a CPU. Resource use (file size,
memory, latency, energy) matters as much as accuracy: a keyboard model runs in the background next to every app.

Non-goals for v1: chat or instruction following, languages other than English, knowledge or reasoning benchmarks,
multi-word suggestions, and on-device training.

Decisions so far:

| Question | Decision |
|---|---|
| Name | Typeling; the model is Typeling-25M (the whole project is renamed accordingly) |
| Approach | the least-effort route: full-parameter continued pretraining of a recent small base model, not training from scratch |
| Base model | `SupraLabs/Supra2-Medium-Base` (25.4M parameters); fallback `SupraLabs/Supra-50M-Base` (section 2) |
| Release files | `ScarletKc/Typeling-25M` on Hugging Face: fp16 safetensors (about 51 MB); `ScarletKc/Typeling-25M-GGUF`: GGUF Q8_0 (about 27 MB, estimate); no 4-bit file: users can quantize the safetensors further with llama.cpp |
| Product | the `typeling` package on PyPI: the n-gram kernel (no dependencies) plus the language model decoder `typeling.lm` (extra `typeling[lm]`: PyTorch, Transformers) and a terminal demo with a candidate bar |
| Data | v1 reuses the existing corpus in `data/corpus/`; more data only if the results show it is the bottleneck |
| Data licenses | permissive only (public domain, CC BY, ODC-By, Apache-2.0); no NC or share-alike data |
| Weights license | Apache-2.0 (the base model's license), with a NOTICE that carries the base model and dataset attributions |
| Typos | repaired in the decoder (keyboard error model, candidates scored by the model); the model is not fine-tuned on typos: a 40M-token typo fine-tuning experiment was worse than decoder repair on both clean and mistyped input (section 14) |
| Candidates | the top 3 are the accuracy target (KSR@3, top3@i); the decoder's search runs until no hypothesis in the beam can overtake the third candidate (not exact: pruned paths are not revisited), further candidates are best effort |
| Hardware | the local RTX 5070 Ti laptop GPU (12 GB); rent a GPU only if a full training run would take more than a few hours |

## 2. Base model

Candidates checked on Hugging Face (quality claims on the community model cards are unverified; zero-shot results on
our benchmark are in section 12):

| Model | Released | Parameters | Architecture | License | Pretraining data | Verdict |
|---|---|---|---|---|---|---|
| SupraLabs/Supra2-Medium-Base | 2026-08 | 25.4M | `Qwen3ForCausalLM` | Apache-2.0 | FineWeb-Edu, 20B tokens | **chosen**: half the parameters of Supra-50M for 1.3–3.5 points of zero-shot KSR |
| SupraLabs/Supra-50M-Base | 2026-05 | 51.8M | `LlamaForCausalLM` | Apache-2.0 | FineWeb-Edu, 20B tokens | fallback: best zero-shot accuracy among the candidates within the size budget |
| SupraLabs/Supra2-100M-Base | 2026-07 | 100.7M | `Qwen3ForCausalLM` | Apache-2.0 | FineWeb-Edu and DCLM, 30B tokens | too large (Q4_K_M about 60–65 MB, estimate) |
| SupraLabs/Supra1.5-50M-Base-exp | 2026-06 | 51.8M | `LlamaForCausalLM` | Apache-2.0 | Supra-50M plus 3B tokens, 60% tool calls and ChatML chats | pushed toward assistant formats |
| SupraLabs Pico, Brain, GDN, Elegans models | 2026-08 to 10 | 0.5–50M | custom (needs remote code) | Apache-2.0 | various | llama.cpp cannot run them |
| harrrshall/BarunLM-35M | 2026-07 | 35.1M | custom (needs remote code) | Apache-2.0 | 5.7B tokens, educational mix | custom architecture |
| MaliosDark/Isabel-50M | 2026-06 | 54.1M | `LlamaForCausalLM` | Apache-2.0 | educational text plus benchmark train splits | not a clean base |
| EleutherAI/pythia-14m, pythia-31m | 2023 | 14.1M, 30.5M | `GPTNeoXForCausalLM` | Apache-2.0 | the Pile (includes Books3) | references only |
| HuggingFaceTB/SmolLM2-135M | 2024-11 | 134.5M | `LlamaForCausalLM` | Apache-2.0 | SmolLM corpus | too large; reference only |

Supra2-Medium-Base, per its model card: an experiment in parameter efficiency, not stronger than Supra-50M but
cheaper. Qwen3 architecture (QK-norm, SwiGLU, RoPE, RMSNorm), hidden size 512, intermediate size 896, 7 layers, 8
attention heads (no GQA), context 1,024, tied embeddings, a 16,384-token byte-level BPE tokenizer (8.4M of the 25.4M
parameters are the embedding matrix). Trained on 20B FineWeb-Edu tokens (about 800 tokens per parameter) with
documents concatenated into one token stream, no separators. It knows nothing specific about chat or informal typing.

Why the smaller model: zero-shot, Supra-50M is 1–2 KSR points better on chat even after simulated 4-bit quantization
(section 12), but costs about twice the compute per token and needs 4 bits to fit the old 37.5 MB budget, while
Supra2-Medium fits at Q8_0 (nearly lossless). Published keyboard work points the same way: the Lexo keyboard measured
Qwen2.5-1.5B against 0.5B for next-word prediction and got 19.0% -> 20.1% top-1 and 39.0% -> 39.4% top-5 for about
2.4x the decoding cost (lexokeyboard.app, "Fitting a language model inside an Android keyboard", 2026). If Typeling
misses the success criteria after milestone 3, the same pipeline is rerun on Supra-50M.

The size argument is weaker than it looks: GGUF Q4_0 stores about 4.5 bits per weight and Q8_0 about 8.5, so a 4-bit
Supra-50M would be about 30 MB (estimate from the parameter count), close to Supra2-Medium at Q8_0 (27.55 MB,
section 14), and Dettmers & Zettlemoyer (2023) find 4 bits with more parameters usually the better trade-off at a
fixed size. The two models were compared only zero-shot, and their speed was not measured.

File sizes are estimates from the parameter count until milestone 3 measures them: f16 about 51 MB, Q8_0 about 27 MB.

## 3. Data

v1 uses the corpus already built for the n-gram model, with its dev and sealed test splits unchanged:

| Source | License | Kind | Training words |
|---|---|---|---|
| FineWeb | ODC-By 1.0 | web writing | 18.5M |
| SODA | CC BY 4.0 | everyday dialogue, generated by InstructGPT | 16.3M |
| WildChat-1M, user turns | ODC-By 1.0 | real typed text | 7.5M |
| Ubuntu IRC | Public Domain | informal internet chat | 4.4M |
| OpenAssistant oasst2, user turns | Apache-2.0 | real typed text | 0.4M |

About 47M words, roughly 65M tokens with the base tokenizer (1.29–1.44 tokens per word, measured). Each chat turn or
document is one segment, separated by the end-of-text token `</s>`. The base model never saw `</s>` or `<s>`
(section 12); `</s>` is therefore initialized from the newline token before training (section 4).

If the evaluation shows that more data would help, the next sources, in order of expected value: the remaining 30% of
WildChat user turns, the rest of SODA, CC BY YouTube transcripts (`common-pile/youtube_filtered`, spoken language),
more Ubuntu IRC, and CC BY news. New data from existing sources uses the same content-hash split, so the dev and test
splits do not change; any training document sharing a 13-gram with the dev or test splits is dropped.

## 4. Training

All stages are full-parameter (LoRA saves nothing meaningful at this size), bf16 mixed precision, AdamW, sequence
packing to 512 tokens, gradient clipping at 1.0.

1. **Continued pretraining.** Two to three epochs over the corpus (about 150–200M tokens), with the learning rate
   re-warmed to a fraction of the base model's peak (3e-3) and decayed to zero. The decay phase weights the typing
   domains (real typed text, SODA, IRC) more heavily. FineWeb stays in the mix so general writing does not degrade.
   - *Segment separator initialization.* Before training, the `</s>` embedding row is set to a copy of the newline
     token's row, a direct edit of the weights, so `</s>` starts out meaning "a new text begins". The embedding is
     tied, so this also makes the model predict `</s>` wherever it would predict a newline; training separates the
     two.
   - *Trailing-space augmentation* (`lm/data.py --space-split`). In 1% of word-initial tokens, `" word"` is
     re-tokenized as `" "` followed by the tokens of `"word"`. The model then also learns to continue text that ends
     in a space, so users of the raw weights (for example llama.cpp without the typing decoder) get sensible
     predictions after a space. The decoder keeps its own trailing-space handling (section 6). The rate was chosen in
     the milestone 1 comparison (section 13): 3% cost about 3% dev perplexity and 0.7 KSR points, 1% costs about 1%
     perplexity and still teaches the behavior.
2. **Typo-robust fine-tuning (experiment, not in v1).** Two special tokens are added (`<|typed|>`, `<|word|>`; the
   embedding matrix is resized). Examples have the form `context <|typed|> noisy-prefix <|word|> intended-word`, where
   the noise model (`lm/typos.py`) applies QWERTY adjacent-key substitutions, omissions, insertions, adjacent
   transpositions, missing apostrophes (dont, im) and casing errors; the context itself gets the same noise on 5% of
   its words. These examples are mixed with clean language-modeling data. The 40M-token run lost 4–12 KSR points on
   clean typing and still did worse on mistyped input than repair in the decoder (section 14), so v1 ships the
   continued-pretraining model unchanged.
3. **Calibration.** Fit a temperature on the dev split so the word probabilities used by display thresholds are
   calibrated.

## 5. Quantization and export

- fp16 `safetensors` plus `config.json` and `tokenizer.json`, loadable with `transformers` as-is, for further
  fine-tuning and conversion to other formats.
- GGUF Q8_0 via llama.cpp's converter (which supports Qwen3): the main release file. Converting it also checks that
  llama.cpp handles the tokenizer.
- No 4-bit file. At 25M parameters, Q4 would save only about 10 MB over Q8_0, which matters on phones rather than
  desktops, and small models lose more to 4 bits (simulated 4-bit quantization cost Supra-50M 0.4–1.1 KSR points
  zero-shot, section 12). Users who want one can quantize the safetensors with llama.cpp; the model card says so.
- The Q8_0 file is evaluated on the benchmark next to the fp16 weights; the model card reports the loss.

Direct weight edits (on the safetensors) are used only for problems that sit in individual token vectors; problems in
the model's behavior (domain, trailing spaces) are fixed by training:

| Edit | Status |
|---|---|
| Initialize `</s>` from the newline token (section 4) | v1 |
| Vocabulary pruning: drop embedding rows of tokens the corpus never uses (code, other scripts) and rebuild the tokenizer, to shrink the GGUF file | optional, only if the size must shrink |
| Vocabulary extension with chat tokens (`" lol"`, `" tbh"`, `" gonna"`), initialized from the mean of their current subword embeddings | not in v1: small gain, requires tokenizer changes |

## 6. Decoding for typing

The model predicts tokens; the user needs words. `typeling/lm.py` (`LMPredictor`) implements this with the same
`topk(text, k)` and `suggest(text)` interface as the n-gram `Predictor`; `lm/decode.py` adds the experimental typed
mode for typo-fine-tuned models.

- Constraint: the text before the cursor is split into the context, the whitespace after it and the partially typed
  word. The context is cut before the trailing whitespace; the whitespace plus the typed letters (" kn") constrain the
  generated text. A prefix phase allows only tokens consistent with the constraint (" k", " kn", " know"); a word
  phase extends the word with letter and contraction tokens until the model ends it. A word's probability is the sum
  over its token paths of path probability times the probability that the next token does not continue the word,
  divided by the probability of the constraint. Paths are explored with a beam (width 8); casings are merged.
- Trailing spaces: with byte-level BPE the space belongs to the next token, so text ending in a space (the usual case
  before a next-word prediction) pushes the model toward junk tokens (section 12); the constraint above avoids it.
- Typo repair (noisy channel): `typeling/keyboard.py` lists every string that one typing error (adjacent key, extra
  key next to a neighbor, left-out letter or apostrophe, swapped letters, missing capital) turns into the typed
  letters. Those that start a word of the n-gram vocabulary are ranked by unigram mass, and the best 4 become extra
  constraints in the same beam, each weighted by the error probability (`typo_weight`, 0.003 relative to no error).
  Completion and correction (tomm -> tomorrow, thnik -> think) use one search, and the number of model calls barely
  changes. Corrections appear in `topk()` and the candidate bar; ghost text shows only extensions.
- Candidates: the search stops once no hypothesis left in the beam can beat the third-best word; more candidates are
  the other words the search reached. The top 3 are not exact: paths pruned from the beam are not revisited (beam 6
  and 8 differ), and hidden words, the user model and n-gram fusion are applied after the search.
- User model: the kernel's user model (words and word pairs the user typed) is mixed into the joint probabilities
  before normalizing over the words that fit the typed letters, with the kernel's weight schedule (up to 0.3), so a
  name the language model barely knows wins after a few uses while common words stay with the language model.
- Context: the last 1,000 characters (section 14).
- N-gram fusion (`ngram_weight`): the final probability mixes in the bundled n-gram model's, so fusion adds at most
  `ngram_weight` to any word. The language model's subword pieces can form non-words (don't + know -> "don'tknow",
  "much" + "for" -> "muchfor"); fusion adds mass to the n-gram model's real words and pushes such non-words down the
  list. Known misspellings from the n-gram model (recieve) are never suggested.
- The context's key/value cache is reused across keystrokes.
- Latency target per suggestion with the main GGUF file on a laptop CPU (normal power mode; power-saving mode is
  reported as the worst case): p95 at most 50 ms, p99 at most 100 ms. 100 ms is about the limit of what feels
  instant and about the gap between keystrokes of a fast typist, so suggestions are computed asynchronously and a
  stale one is dropped when the next key arrives; the two-tier integration (milestone 6) shows the n-gram suggestion
  at once. Phones are several times slower than a laptop CPU, so a mobile build needs a tighter budget.
- Later option: a word-scoring head that scores whole candidate words in one forward pass instead of decoding tokens,
  the design the Lexo keyboard ships; it would cut the per-suggestion cost to one model call.

## 7. Evaluation

- Benchmark: `bench/eval.py`'s typing simulation on the dev splits (chat, casual, writing), with the n-gram model's
  metrics: KSR (ghost text), precision, suggestions shown, KSR@3 (a bar of 3 candidates, always shown), acc@0–3,
  top3@0–3, latency (mean and p99). `lm/eval.py` runs it on any causal LM through the decoder, plus per-word
  perplexity and tokens per word. For speed it defaults to 2,000 words per source; compare against the n-gram model
  at the same `--limit`.
- Resource use: file size, peak memory and per-suggestion latency on a laptop CPU with llama.cpp (milestone 3).
- Baselines on the same benchmark: the strongest n-gram model the corpus allows (no pruning, 23.35 MB; built with
  `uv run training/build.py model --out data/ngram-unpruned.bin --budget 30e6`), the bundled 5 MB n-gram model,
  the base model before fine-tuning, Supra-50M, Pythia-14M and Pythia-31M (references only).
- Robustness: `lm/eval.py --typos` types every word of 3+ letters with a synthetic error and compares KSR and KSR@3
  with clean typing on the same words (section 14).
- Success criteria for v1 (targets to validate, not promises): with the main GGUF file on chat, at equal precision,
  KSR at least 3 points and acc@0 at least 5 points above the unpruned 23 MB n-gram model; latency target met.
- The test splits stay sealed until the final comparison.

## 8. Compute

Training cost is about 6 × parameters × tokens: about 2.3e16 FLOPs for 25.4M parameters and 150M tokens. Measured
in the milestone 1 pilot: about 60k tokens/s on the RTX 5070 Ti laptop GPU (97% utilization, 9.5 GB of memory at 32 ×
512 tokens per micro-batch), so 150–200M tokens take 40–55 minutes; typo fine-tuning is smaller. PyTorch
with CUDA 12.8 or newer is required for this GPU (Blackwell); it is installed in an isolated uv environment, never as
a dependency of the runtime package.

## 9. Milestones

0. **Setup and gate** (done, section 12). PyTorch environment, typing decoder, benchmark adapter, zero-shot
   evaluation of the candidates, trailing-space and tokenization checks, choice of the base model.
1. **Continued pretraining** (done, section 13) on the existing corpus, after a short pilot, and a CPU latency check
   of the decoder in PyTorch.
2. **Typos and the product** (done, section 14): typo benchmark, decoder typo repair, the typo fine-tuning experiment
   (not adopted), the decoder packaged as `typeling.lm` with a candidate bar and n-gram fusion, demo and tests.
3. GGUF export (Q8_0), final evaluation on the test split against the baselines, CPU latency and memory.
4. Release: `ScarletKc/Typeling-25M` (fp16 safetensors, model card, NOTICE), `ScarletKc/Typeling-25M-GGUF` (Q8_0),
   the GitHub repository made public, and the `typeling` package on PyPI.
5. Optional: more data (section 3), if milestone 3 shows the corpus is the bottleneck.
6. Optional: asynchronous two tiers like the Lexo keyboard: the n-gram model and the user model answer instantly,
   and the language model's ranking replaces them when it arrives (v1 fuses both synchronously); a llama.cpp port of
   the decoder; typo fine-tuning only if a better recipe closes the remaining gap without hurting clean typing.

Fallbacks if the fine-tuned base model disappoints: the same pipeline on Supra-50M-Base; failing that, train a
15M-parameter model from scratch with a 16k BPE tokenizer on an expanded corpus (an earlier version of this plan,
kept in git history).

## 10. Risks

- **Base model maturity.** Supra2-Medium-Base comes from a small lab, is labeled an efficiency experiment, has few
  downloads and unverified benchmark claims; milestone 0 checked it on our benchmark.
- **Capacity.** It is 1.3–3.5 KSR points behind Supra-50M zero-shot; continued pretraining has to open a 3-point lead over
  the unpruned n-gram model on chat. Fallback: Supra-50M.
- **Domain shift.** The base model saw only educational web text; continued pretraining has to move it toward chat
  and informal typing, and the corpus is small.
- **Base tokenizer** splits chat slang into more pieces than Supra-50M's (`" g" "on" "na"`), which costs decoding
  steps and some of the speed advantage.
- **Contamination of the writing domain.** FineWeb-Edu is filtered from FineWeb, so the base model may have seen some
  FineWeb dev documents; writing-domain results may be optimistic and the model card must say so.
- **SODA's origin.** It was generated by InstructGPT; OpenAI's terms restrict using outputs to build competing models.
  The risk for a typing model is small, but the model card must disclose it, and an ablation without SODA shows what
  it contributes.
- **Subword decoding** adds latency and complexity compared with the n-gram's word-level scoring; the latency target
  is the check, and a word-scoring head (section 6) the fallback.
- **Memorization and PII**: the corpus is already scrubbed of emails, URLs and phone numbers; test for verbatim
  reproduction of training text before release. Not done for v1; the technical report says so.

## 11. Repository layout

```text
typeling/
  lm.py          typing decoder: LMPredictor (typo repair, candidates, n-gram fusion)  implemented
  keyboard.py    QWERTY error model used by the typo repair                           implemented
lm/
  decode.py      experimental typed mode for typo-fine-tuned models                   implemented
  eval.py        typing and typo benchmarks, perplexity, simulated Q4                 implemented
  data.py        tokenize data/corpus into packed token streams                       implemented
  typos.py       typo simulation and typo fine-tuning data (experiment)               implemented
  train.py       continued pretraining and typo fine-tuning (PyTorch)                 implemented
  export.py      fp16 safetensors, GGUF, model cards
```

Training dependencies (torch, transformers, gguf) are declared as uv inline script metadata. The runtime package
stays dependency-free except for `typeling.lm`, which needs the optional extra `typeling[lm]` and is imported by
nothing else in the package.

## 12. Milestone 0 results

Environment: PyTorch 2.11 (CUDA 12.8) and transformers 5.22 on the RTX 5070 Ti laptop GPU; evaluation in float32.

**Typing benchmark, zero-shot.** Dev split, 2,000 words per source, display thresholds 0.3 / 0.3. KSR by domain
(chat = SODA + WildChat + oasst2):

| Model | Parameters | Chat | Casual (IRC) | Writing (FineWeb) |
|---|---|---|---|---|
| n-gram, bundled | 5 MB | 29.67% | 22.81% | 23.54% |
| n-gram, unpruned | 23 MB | 31.86% | **24.82%** | 25.49% |
| Pythia-14M | 14.1M | 28.51% | 21.75% | 33.56% |
| Pythia-31M | 30.5M | 31.17% | 22.91% | 36.55% |
| **Supra2-Medium-Base** | 25.4M | 33.47% | 22.84% | 38.58% |
| Supra-50M-Base, simulated Q4_0 | 51.8M | 35.03% | 23.44% | 41.04% |
| Supra-50M-Base | 51.8M | **35.62%** | 24.11% | **42.13%** |

Per source, KSR / acc@0:

| Model | SODA | WildChat | oasst2 | IRC | FineWeb |
|---|---|---|---|---|---|
| n-gram, unpruned | **38.55% / 30.60%** | 28.87% / 17.95% | 29.04% / 19.30% | **24.82%** / 13.95% | 25.49% / 16.50% |
| Supra2-Medium-Base | 33.96% / 26.00% | 30.98% / 20.60% | 35.65% / 25.15% | 22.84% / 13.30% | 38.58% / 27.85% |
| Supra-50M-Base, simulated Q4_0 | 36.27% / 28.00% | 32.59% / 21.80% | 36.49% / 26.40% | 23.44% / 15.20% | 41.04% / 28.85% |
| Supra-50M-Base | 36.90% / 28.80% | **32.95% / 23.15%** | **37.29% / 26.55%** | 24.11% / **15.40%** | **42.13% / 29.60%** |

- Without any training on our data, both Supra models beat the unpruned n-gram model on real typed text (WildChat,
  oasst2) and lose on SODA (whose style the n-gram model memorized from 16M training words) and slightly on IRC.
- The writing domain favors every LM by a wide margin, partly from long-range context and possibly from
  contamination (section 10).
- Simulated 4-bit quantization (`lm/eval.py --simulate-q4`: linear layers rounded as GGUF Q4_0, the embedding matrix
  at 6 bits; pessimistic compared with Q4_K_M) costs Supra-50M 0.4–1.1 KSR points and raises per-word perplexity by
  13–20%. Quantized Supra-50M stays 0.4–2.5 points ahead of Supra2-Medium in full precision.
- Latency on the GPU in this harness: 12–18 ms per query for Supra2-Medium, 16–26 ms for Supra-50M (some runs shared
  the GPU), dominated by per-step overhead; CPU latency with llama.cpp is measured in milestone 3.
- The Supra2-Medium rows come from a rerun after a decoder fix: its tokenizer adds a space before every text, so the
  first word of a text must be decoded as " word". The fix changed its results by at most 0.2 points.

**Start marker.** Neither Supra model was trained with document separators, so `<s>` and `</s>` are untrained. Per-word
perplexity on the dev splits by the text placed before each segment:

| Model, start marker | SODA | WildChat | oasst2 | IRC | FineWeb |
|---|---|---|---|---|---|
| Supra-50M, `<s>` or `</s>` (identical results) | 95.9 | 663.9 | 207.2 | 2,174 | 103.4 |
| Supra-50M, newline | 68.6 | 493.2 | 146.4 | 1,405 | 101.8 |
| Supra-50M, two newlines | 72.7 | | | | |
| Supra2-Medium, `<s>` | 152.9 | 966.3 | 325.8 | 2,845 | 145.1 |
| Supra2-Medium, newline | 131.9 | 805.4 | 250.4 | 2,697 | 141.3 |

Zero-shot evaluation uses a newline; training uses `</s>`, initialized from the newline token (section 4). The real
typed text (WildChat, IRC) is far from the base models' domain; this is what continued pretraining targets.

**Trailing space.** Confirmed on both models: on text ending in a space, the raw next-token distribution is junk
(Supra-50M: `"Let me know "` -> `'....'` 0.23, `'___'` 0.09, `'!!'` 0.08; Supra2-Medium: `'!'` 0.37, an invalid
byte 0.19, `'!!'` 0.08), while the decoder returns sensible words (Supra2-Medium: if 0.48, what 0.11, how 0.07).

**Tokenization.** Tokens per word on the dev splits: Supra-50M 1.23–1.35, Supra2-Medium 1.29–1.44. Both keep common
words and contractions compact (`" i"`, `" I" "'m"`, `" don" "'t"`, `" ok"`, `" tomorrow"`) and split chat slang
(`" l" "ol"`, `" id" "k"`, `" t" "b" "h"`, `" ye" "ah"`); "gonna" is one token for Supra-50M and three for
Supra2-Medium. No tokenizer changes in v1.

**How far the n-gram model scales.** Without any pruning the n-gram model is 23.35 MB (1.66M bigrams, 3.41M
trigrams): the corpus cannot fill 30 MB. Dev split, 20,000 words per source:

| Domain | KSR, 5 MB | KSR, 23 MB | Precision, 5 MB -> 23 MB |
|---|---|---|---|
| chat | 32.11% | 34.31% | 51.5% -> 51.6% |
| casual | 22.81% | 24.58% | 43.8% -> 43.3% |
| writing | 23.11% | 25.00% | 43.2% -> 42.1% |

The gain comes almost entirely from completing partial words; next-word accuracy (acc@0) barely moves: SODA 28.63%
-> 29.16%, WildChat 21.26% -> 22.60%, IRC 13.84% -> 14.41%, FineWeb 14.26% -> 14.56%. A trigram model sees only
the previous two words, so more size only adds rarer word combinations. The unpruned model is therefore the baseline
for the success criteria (section 7).

## 13. Milestone 1 results

**Pilot.** `uv run lm/train.py --model SupraLabs/Supra2-Medium-Base --tokens 20e6 --out data/lm/runs/pilot-medium`:
20M tokens (611 steps of 32,768), peak learning rate 6e-4, mix FineWeb 0.30, SODA 0.25, WildChat 0.25, IRC 0.15,
oasst2 0.05, no trailing-space augmentation; 5.6 minutes at 59k tokens/s.

Dev loss per token, before -> after: FineWeb 3.745 -> 3.762, SODA 2.848 -> 1.929, WildChat 4.494 -> 3.775, IRC
4.722 -> 3.985, oasst2 3.767 -> 3.390 (part of the drop is the model learning the `</s>` separator). Still falling at
the end of the decay.

Typing benchmark (dev split, 2,000 words per source, thresholds 0.3 / 0.3, float32), KSR by domain:

| Model | Chat | Casual (IRC) | Writing (FineWeb) |
|---|---|---|---|
| n-gram, unpruned (23 MB) | 31.86% (precision 50.5%) | 24.82% (43.9%) | 25.49% (41.5%) |
| Supra2-Medium-Base, zero-shot | 33.47% (49.9%) | 22.84% (42.8%) | 38.58% (48.5%) |
| **Pilot** | **40.30% (52.6%)** | **27.74% (44.8%)** | 37.48% (47.4%) |

Per source, pilot KSR / acc@0 (change in acc@0 against the unpruned n-gram model): SODA 46.01% / 41.65% (+11.1),
WildChat 36.34% / 24.75% (+6.8), oasst2 39.36% / 28.10% (+8.8), IRC 27.74% / 17.80% (+3.9), FineWeb 37.48% /
26.85% (+10.4). Per-word perplexity: SODA 24.7, WildChat 228, oasst2 109, IRC 408, FineWeb 159 (zero-shot: 132, 805,
250, 2,697, 141).

- In full precision, the pilot already meets the v1 success criteria on chat: KSR 8.4 points above the unpruned
  n-gram model at higher precision, and acc@0 6.8–11.1 points higher on every chat source. Still to show: the same
  after GGUF quantization, and on the sealed test split.
- Writing regresses slightly (KSR −1.1 points, perplexity +13%) as the model moves toward chat; it stays far above the
  n-gram model.
- SODA gains the most; it is synthetic and formulaic, so WildChat, oasst2 and IRC are the more telling numbers.

**Learning rate and trailing-space augmentation.** Same setup as the pilot (20M tokens, same mix); dev loss per token
at the end:

| Run | FineWeb | SODA | WildChat | IRC | oasst2 |
|---|---|---|---|---|---|
| lr 3e-4 | **3.714** | 1.984 | 3.830 | 4.033 | **3.382** |
| lr 6e-4 (the pilot) | 3.762 | 1.929 | 3.775 | 3.985 | 3.390 |
| lr 1.2e-3 | 3.862 | **1.907** | **3.768** | **3.965** | 3.433 |
| lr 6e-4, space split 3% | 3.781 | 1.964 | 3.813 | 4.016 | 3.418 |
| lr 6e-4, space split 1% | 3.770 | 1.939 | 3.800 | 3.996 | 3.398 |

- A higher learning rate adapts faster to chat and forgets more writing; 1.2e-3 gains 0.01–0.02 on chat for 0.1 on
  FineWeb, so 6e-4 is kept.
- The augmentation works: on `"Let me know "` the raw model without it puts its mass on junk (`'!'` 0.24, an invalid
  byte 0.23); with 3% it predicts `'if'` 0.69, `'how'` 0.16, `'what'` 0.10, with 1% `'what'` 0.33, `'if'` 0.32,
  `'how'` 0.29. The decoder's suggestions barely change.
- Its cost: 3% raises dev loss by about 0.03 on every source and costs about 0.7 KSR points (SODA 45.38%, WildChat
  35.61%, against 46.01% and 36.34% for the pilot); 1% raises dev loss by about 0.01. The full run uses 1%.

**Full continued pretraining.** `uv run lm/train.py --model SupraLabs/Supra2-Medium-Base --tokens 150e6 --lr 6e-4
--mix "fineweb=0.3,soda=0.25,wildchat=0.27,irc=0.17,oasst2=0.01" --data
data/lm/tokens/SupraLabs--Supra2-Medium-Base+space0.01 --eval-every 250 --save-every 1000 --out data/lm/runs/cpt-150m`:
150M tokens (4,578 steps, about 2.2 epochs of the corpus), lr 6e-4, oasst2 down to 0.01 (it has only 0.5M tokens; at
0.05 it would be repeated about 15 times), space split 1%. 57.9 minutes at 43k tokens/s on average: the laptop ran in
power-saving mode for most of it (GPU clock about 920 MHz instead of up to 3,090 MHz), at full speed it is about 60k.

Dev loss per token, before -> pilot -> full run: FineWeb 3.745 -> 3.762 -> 3.793, SODA 2.848 -> 1.929 -> 1.711,
WildChat 4.494 -> 3.775 -> 3.522, IRC 4.722 -> 3.985 -> 3.821, oasst2 3.767 -> 3.390 -> 3.434; flat over the last
500 steps.

Typing benchmark (same setup as above), KSR (precision):

| Model | Chat | Casual (IRC) | Writing (FineWeb) |
|---|---|---|---|
| n-gram, unpruned (23 MB) | 31.86% (50.5%) | 24.82% (43.9%) | 25.49% (41.5%) |
| Pilot, 20M tokens | 40.30% (52.6%) | 27.74% (44.8%) | **37.48%** (47.4%) |
| **Full run, 150M tokens** | **41.16% (52.3%)** | **29.60% (44.8%)** | 37.16% (47.1%) |

Per source, KSR / acc@0 (change in acc@0 against the unpruned n-gram model): SODA 48.00% / 45.45% (+14.9), WildChat
38.10% / 26.65% (+8.7), oasst2 38.28% / 27.60% (+8.3), IRC 29.60% / 19.55% (+5.6), FineWeb 37.16% / 26.85% (+10.4).
Per-word perplexity: SODA 19.1, WildChat 156, oasst2 115, IRC 312, FineWeb 176.

- Against the pilot, the 7.5x longer run adds 0.9 KSR points on chat and 1.9 on casual text (WildChat +1.8, IRC
  +1.9); oasst2 loses 1.1 points after its weight was cut, writing is flat. Returns are diminishing on this corpus;
  more data (section 3) is the likelier next gain than more epochs.
- With 2,000 words per source, differences of about a point are within sampling noise (an estimate, not measured:
  samples of this size can differ by several points, section 15).

**CPU latency, upper bound.** The full-run model through the real decoder in PyTorch (float32, eager) on the laptop
CPU with 4 threads, 300 words per source (`lm/eval.py --device cpu --threads 4 --no-ppl --limit 300`), per query
mean (p95 / p99): SODA 21.2 ms (33.9 / 41.8), WildChat 18.4 ms (30.9 / 39.3), IRC 18.4 ms (34.0 / 46.8), FineWeb
27.5 ms (53.3 / 76.9). Short texts meet the target (p95 ≤ 50 ms, p99 ≤ 100 ms) with room to spare, before any
llama.cpp speedup. Long documents are slower because the per-query cost grows with the context: 15 ms with 30 tokens
of context, 18 ms with 140, 19 ms with 250 and 25 ms with 480 (beam 8, same setup). In this PyTorch decoder every
beam step copies the whole key/value cache for each hypothesis (`reorder_cache`); llama.cpp can share the context's
cache between sequences. Separately, texts longer than 2,000 characters lost the cache on every word, because the
context was cut at exactly the last 2,000 characters; the cut point now moves in steps of 500 characters
(`CONTEXT_STEP`). That case is not part of these measurements (300 words stay within the first 2,000 characters).

## 14. Milestone 2 results: typos

Typo benchmark (`lm/eval.py --typos --limit 1000`, dev split, 1,000 words per source): every word of 3+ letters is
typed with a synthetic error (one, two with probability 0.2; 76–82% of all words), and the KSR is compared with clean
typing of the same words. Ghost-text KSR, clean -> with typos (share of mistyped words that still end correct):

| Model, decoder | Chat | Casual (IRC) | Writing (FineWeb) |
|---|---|---|---|
| n-gram, bundled 5 MB | 29.41% -> 20.39% (48.4%) | 21.62% -> 14.28% (39.2%) | 22.73% -> 14.67% (39.2%) |
| cpt-150m, no repair | 41.34% -> 32.53% (61.7%) | 28.79% -> 20.20% (47.9%) | 34.71% -> 26.34% (54.3%) |
| **cpt-150m, repair 0.003** | 41.13% -> **36.53%** (84.3%) | 28.33% -> **22.82%** (72.1%) | 34.62% -> **29.60%** (82.0%) |
| cpt-150m, repair 0.01 | 40.97% -> 36.70% (85.1%) | 28.05% -> 22.82% (73.7%) | 34.30% -> 29.69% (82.0%) |
| cpt-150m, repair 0.03 | 40.67% -> 36.65% (85.5%) | 27.59% -> 22.73% (74.7%) | 33.58% -> 29.60% (82.8%) |
| typo fine-tuned, typed mode | 33.98% -> 28.78% (73.9%) | 22.58% -> 16.73% (62.5%) | 24.52% -> 19.10% (61.4%) |

- Typing every word with an error costs the plain decoder 8.4–8.8 KSR points. Decoder repair (weight 0.003, at most
  4 corrections, from 2 typed letters) recovers 36–48% of that (the drop shrinks to 4.6 points on chat, 5.5 casual,
  5.0 writing) at a clean cost of 0.1–0.5 points, and the share of mistyped words that end correct rises from 48–62%
  to 72–84%.
- The error weight hardly matters on mistyped input between 0.003 and 0.03; higher weights cost more on clean
  input, so the default is 0.003.
- What repair cannot reach: words with two errors (one in five mistyped words here), and errors whose correction is
  not a prefix of a vocabulary word.
- At a realistic share r of mistyped words, KSR is about (1 - r) x clean + r x with typos: at r = 10%, 40.67% with
  repair against 40.46% without on chat (+0.21 points; casual -0.15, writing +0.25). The visible gain is in
  corrections, not in raw keystrokes.
- The benchmark is optimistic: the errors use `typeling.keyboard`'s layout and error types, the ones the repair
  inverts; context words are never mistyped; Correct also counts words accepted before the error (hence 48–62%
  without repair); and `score()` counts the top candidate as accepted ghost text even when it is a correction that
  does not extend the typed letters, which `suggest()` never shows. With typos, ghost-text KSR and Correct are
  therefore upper bounds for `suggest()`; KSR@3 and Correct@3 match `topk()`.

**Typo fine-tuning experiment.** `uv run lm/typos.py --model data/lm/runs/cpt-150m --out data/lm/tokens/typo` (600k
examples, 47% with errors), then 40M tokens from cpt-150m (1,221 steps, lr 3e-4, half typo examples with the loss on
the intended word only, half the usual corpus mix), decoded in typed mode (`lm/decode.py`). The typo dev loss fell
from 5.78 to 1.14, but the clean dev loss stayed 0.07–0.12 above cpt-150m (FineWeb 3.79 -> 3.90, WildChat 3.52 ->
3.64), and the typing benchmark is worse in every column: clean KSR 6–10 points below cpt-150m with repair, mistyped
KSR 6–11 points below; on SODA, its acc@full (the intended word on top after the whole mistyped word) was 84%. Typos
stay in the decoder; the model is released without typo fine-tuning.

**Candidate bar, n-gram fusion, unknown prefixes.** Same benchmark, typo weight 0.003; ghost-text KSR and KSR@3 (bar
of 3 candidates), clean / with typos, and the precision of ghost text with typos:

| Decoder | Chat KSR | Chat KSR@3 | Casual KSR | Casual KSR@3 | Writing KSR | Writing KSR@3 | Precision (typos), chat |
|---|---|---|---|---|---|---|---|
| n-gram weight 0 | 41.13 / 36.53 | 56.51 / 52.66 | 28.33 / 22.82 | 45.28 / 39.70 | 34.62 / 29.60 | 50.53 / 45.64 | 46.1% |
| n-gram weight 0.1 | 41.18 / 36.54 | 56.76 / 52.77 | 27.95 / 22.39 | 45.57 / 40.03 | 34.38 / 29.15 | 50.75 / 45.90 | 47.7% |
| n-gram weight 0.2 | 40.84 / 36.02 | 56.54 / 52.45 | 27.59 / 22.08 | 45.62 / 40.18 | 33.97 / 28.47 | 50.84 / 45.73 | 49.2% |
| n-gram weight 0.3 | 40.35 / 35.35 | 56.34 / 52.34 | 27.37 / 21.79 | 45.28 / 39.72 | 33.47 / 27.89 | 51.01 / 45.56 | 50.3% |
| **n-gram 0.1, unknown weight 0.1** | 41.18 / 36.66 | 56.76 / 52.91 | 27.95 / 22.51 | 45.55 / 40.30 | 34.38 / 29.38 | 50.75 / 45.97 | 47.8% |

- A bar of 3 candidates saves 15–17 points more than ghost text alone (KSR@3 56.8% on chat), and the intended word
  is among the top 3 for 87–94% of mistyped words at some point while typing them (Correct@3).
- N-gram fusion at 0.1 adds 0.2–0.3 points of KSR@3 and 1–2 points of precision, at most 0.4 points of ghost-text
  KSR; higher weights cost more. The visible gain is outside the metric: fewer non-words among candidates 4–5.
  (An earlier version of this note said that encyc -> encyclopedia, 0.93 with fusion, was absent without it. That
  cannot be: at weight 0.1, fusion adds at most 0.1, so the language model alone gives encyclopedia at least 0.92.)
- Lowering the weight of typed letters that start no vocabulary word to 0.1 changes nothing on clean input and
  adds 0.1–0.3 points with typos; "I will recie" now ranks receive first (the model had learned "recieve").
- Defaults: typo weight 0.003, unknown weight 0.1, n-gram weight 0.1.

**User model.** `LMPredictor(user=path)` learns with `learn()` and stores the same file as the n-gram kernel; the
user model is mixed into the joint probabilities (section 6). After one sentence is learned three times, "I love Zorb"
ranks "Zorblaxa" first, and common completions do not change (`tests/test_lm.py`). The dev benchmark with `--learn`
shows no gain (chat KSR 41.19% -> 40.88%, KSR@3 56.76% -> 56.65%; the n-gram kernel moves by at most 0.4 points
too): its texts come from thousands of different people, so there is little personal vocabulary to learn. Measuring
it would take text written by one person.

**Context length and beam** (typing benchmark, 1,000 words per source). With 1,000 characters of context instead of
2,000, chat and IRC are unchanged (their texts are shorter) and FineWeb loses 0.1 points (KSR 34.28% vs 34.38%,
KSR@3 50.68% vs 50.75%), while its CPU time drops from 36 ms mean (76 / 110 ms p95 / p99) to 33 ms (63 / 82 ms).
Beam 6 is 0.1–0.6 points worse than beam 8 and not faster. Defaults: 1,000 characters, beam 8.

**CPU latency, v1 decoder** (typo repair and n-gram fusion on, PyTorch float32, 4 threads, 300 words per source),
mean (p95 / p99): SODA 24.8 ms (39.1 / 51.1), WildChat 23.7 ms (39.9 / 52.0), oasst2 23.5 ms (44.7 / 56.0), IRC
23.4 ms (47.7 / 59.7), FineWeb 32.8 ms (63.0 / 82.2). Chat and casual text meet the target (p95 ≤ 50 ms, p99 ≤ 100
ms); long writing meets p99 but not p95. Most of the time goes into copying the context's key/value cache for every
hypothesis at every beam step (`reorder_cache`, then `torch.cat` when the cache grows); a decoder on llama.cpp, whose
sequences can share the context's cache cells, would remove it.

**Export checks.** `lm/export.py`: fp16 safetensors, 50.75 MB, with the same dev perplexity as float32 on every
source (SODA 19.1, WildChat 156.0, oasst2 115.3, IRC 311.5, FineWeb 176.2). GGUF Q8_0, 27.55 MB (llama.cpp b11953),
against an F16 GGUF on 380k characters of dev text (190 chunks of 512 tokens, `llama-perplexity --kl-divergence`):
perplexity 30.025 vs 30.000 (+0.08%), mean KL divergence 0.00094, the same top token 97.9% of the time. The converter
does not know the tokenizer's pre-tokenizer hash; `lm/export.py` maps it to GPT-2's (the same regex). llama.cpp then
tokenizes exactly like Transformers, except that it does not add the space Transformers puts before every text, and
it adds no BOS token: `"</s> Thank you so"` -> `[2, " Thank", " you", " so"]`, the decoder's own input.
The exported tokenizer config names `PreTrainedTokenizerFast` and sets `add_prefix_space`, so Transformers 4.57
loads it and tokenizes like 5.22 (Transformers 5 writes `TokenizersBackend`, unknown to 4.x, and 4.x otherwise
drops the prefix space).

## 15. Milestone 3 results: final evaluation

Sealed test split, run once with the released files (`data/release/Typeling-25M`, fp16) and the v1 decoder defaults
(typo weight 0.003, unknown weight 0.1, n-gram weight 0.1, 1,000 characters of context, beam 8); 2,000 words per
source for typing, 1,000 for typos. Commands: `uv run lm/eval.py --model data/release/Typeling-25M --final --limit
2000`, the same with `--typos --limit 1000`, `python bench/eval.py --final --limit 2000` (with `--model
data/ngram-unpruned.bin` for 23 MB) and `uv run lm/eval.py --ngram --final --typos --limit 1000`.

| Domain | n-gram 5 MB: KSR / KSR@3 | n-gram 23 MB: KSR / KSR@3 | Typeling-25M: KSR / KSR@3 | Precision, 23 MB -> Typeling |
|---|---|---|---|---|
| Chat | 31.58% / 48.78% | 34.08% / 50.64% | **43.12% / 57.99%** | 51.2% -> 54.7% |
| Casual (IRC) | 22.69% / 38.85% | 24.42% / 40.36% | **29.75% / 46.83%** | 42.7% -> 47.0% |
| Writing (FineWeb) | 22.30% / 40.56% | 23.86% / 41.46% | **32.40% / 49.94%** | 44.1% -> 49.7% |

acc@0 / top3@0, n-gram 23 MB -> Typeling-25M: SODA 30.00% / 46.90% -> 43.40% / 62.50%, WildChat 22.55% / 35.50% ->
31.25% / 45.85%, oasst2 18.90% / 30.65% -> 28.10% / 44.50%, IRC 13.85% / 23.95% -> 18.95% / 32.40%, FineWeb 15.00% /
27.10% -> 24.80% / 37.70%. Per-word perplexity: SODA 17.7, WildChat 180.2, oasst2 107.2, IRC 322.6, FineWeb 174.9.

Typos (every word of 3+ letters mistyped), KSR clean -> with typos (Correct), ghost text and bar:

| Domain | n-gram 5 MB | Typeling-25M |
|---|---|---|
| Chat | 32.05% -> 22.71% (48.5%); bar 49.60% -> 40.52% (71.0%) | 44.45% -> 40.07% (85.6%); bar 59.40% -> 55.69% (94.5%) |
| Casual | 22.47% -> 14.82% (36.9%); bar 39.14% -> 29.26% (57.8%) | 28.64% -> 23.09% (73.1%); bar 46.54% -> 41.30% (88.4%) |
| Writing | 24.34% -> 15.70% (40.9%); bar 42.26% -> 32.02% (62.9%) | 33.65% -> 27.91% (78.5%); bar 50.82% -> 45.51% (90.5%) |

- **Success criteria met** (section 7): on chat, KSR is 9.0 points above the unpruned 23 MB n-gram model at higher
  precision (target: 3), and acc@0 is 8.7–13.4 points higher on each chat source (target: 5). The numbers are for the
  fp16 weights through the PyTorch decoder; the Q8_0 GGUF is within 0.1% perplexity (section 14) but has not been run
  through a word-level decoder.
- **Latency target** met on chat and casual text, p99 met on long writing, p95 missed there (63 ms, section 14).
  llama.cpp on the Q8_0 file, 4 threads: 11,900 tokens/s prompt processing (pp256), 2,800 tokens/s single-token
  generation (`llama-bench -t 4 -p 64,256 -n 16`).
- Dev and test agree within about a point per domain, except WildChat (dev KSR 37.15% at 1,000 words, test 42.16%
  at 2,000), where the test sample happens to be easier for both models (the 23 MB n-gram model goes from 28.9% on
  dev to 34.8% on test).
- `--limit` takes the first N words of each split, so the writing samples cover few documents: FineWeb's first
  2,000 test words come from three documents, the first 1,000 (typos) from one, and the 300-word dev latency sample
  from one. No confidence intervals, paired bootstrap or repeated runs on other samples.
- **Footprint** (typeling 0.1.0 installed from PyPI, CPU PyTorch, the same laptop): `import torch` 4.6 s;
  `LMPredictor()` 2.5 s from the Hugging Face cache offline, 9.3 s and 12.6 s in two runs with Hugging Face's online
  revision check (issue #1); resident memory 549 MB after loading. The n-gram kernel: `Predictor()` 0.03 s, 54 MB.
  llama.cpp's CPU binaries (b11953, all DLLs and one executable) take 50 MB on disk.
