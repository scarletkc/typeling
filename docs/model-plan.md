# Plan: Typeling-25M, a small language model for typing prediction

Status: milestone 0 done (section 12): the typing decoder (`lm/decode.py`) and the benchmark adapter (`lm/eval.py`)
are implemented, the candidates are evaluated, and the base model is chosen. Milestone 1 in progress: the data
pipeline (`lm/data.py`) and training script (`lm/train.py`) are implemented, and a 20M-token pilot already meets the
v1 success criteria in full precision (section 13).

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
| Release files | fp16 safetensors (about 51 MB) and GGUF Q8_0 (about 27 MB, estimate); also GGUF Q4_K_M (about 18 MB, estimate) if milestone 3 shows its loss is small |
| Data | v1 reuses the existing corpus in `data/corpus/`; more data only if the results show it is the bottleneck |
| Data licenses | permissive only (public domain, CC BY, ODC-By, Apache-2.0); no NC or share-alike data |
| Weights license | Apache-2.0 (the base model's license), with a NOTICE that carries the base model and dataset attributions |
| Typo-robust fine-tuning | included in v1 |
| Hardware | the local RTX 5070 Ti laptop GPU (12 GB); rent a GPU only if a full training run would take more than a few hours |

## 2. Base model

Candidates checked on Hugging Face (quality claims on the community model cards are unverified; zero-shot results on
our benchmark are in section 12):

| Model | Released | Parameters | Architecture | License | Pretraining data | Verdict |
|---|---|---|---|---|---|---|
| SupraLabs/Supra2-Medium-Base | 2026-08 | 25.4M | `Qwen3ForCausalLM` | Apache-2.0 | FineWeb-Edu, 20B tokens | **chosen**: half the compute and file size of Supra-50M for 1–2 points of KSR |
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

File sizes are estimates from the parameter count until milestone 3 measures them: f16 about 51 MB, Q8_0 about 27 MB,
Q4_K_M about 18 MB (llama.cpp keeps the tied embedding matrix, which is also the output tensor, at 6 bits, and the
MLP intermediate size of 896 is not a multiple of 256, so the MLP down-projection falls back from K-quants).

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
2. **Typo-robust fine-tuning.** Two special tokens are added (`<|typed|>`, `<|word|>`; the embedding matrix is
   resized). Examples have the form `context <|typed|> noisy-prefix <|word|> intended-word`, where the noise model
   applies QWERTY adjacent-key substitutions, omissions, insertions, adjacent transpositions, missing apostrophes
   (dont, im) and casing errors; the context itself gets the same noise on about 5–10% of its words. These examples
   are mixed with clean language-modeling data so ordinary prediction does not degrade.
3. **Calibration.** Fit a temperature on the dev split so the word probabilities used by display thresholds are
   calibrated.

## 5. Quantization and export

- fp16 `safetensors` plus `config.json` and `tokenizer.json`, loadable with `transformers` as-is, for further
  fine-tuning and conversion to other formats.
- GGUF Q8_0 via llama.cpp's converter (which supports Qwen3): the main release file.
- GGUF Q4_K_M as a smaller variant, only if milestone 3 shows a small loss (simulated 4-bit quantization of Supra-50M
  cost 0.4–1.1 KSR points zero-shot, section 12; a smaller model may lose more).
- Every GGUF file is evaluated on the benchmark next to the fp16 weights; the model card reports the loss.
- Only if the Q4 loss is significant and a 4-bit file is still wanted: a short quantization-aware training phase at
  the end of training that simulates GGUF Q4_0 block quantization (blocks of 32 weights with one scale each) in the
  forward pass, followed by a Q4_0 export. This is the approach of Google's QAT releases of Gemma 3 for llama.cpp.

Direct weight edits (on the safetensors) are used only for problems that sit in individual token vectors; problems in
the model's behavior (domain, trailing spaces) are fixed by training:

| Edit | Status |
|---|---|
| Initialize `</s>` from the newline token (section 4) | v1 |
| Vocabulary pruning: drop embedding rows of tokens the corpus never uses (code, other scripts) and rebuild the tokenizer, to shrink the GGUF file | optional, only if the size must shrink |
| Vocabulary extension with chat tokens (`" lol"`, `" tbh"`, `" gonna"`), initialized from the mean of their current subword embeddings | not in v1: small gain, requires tokenizer changes |

## 6. Decoding for typing

The model predicts tokens; the user needs words. `lm/decode.py` implements this with the same `topk(text, k)` and
`suggest(text)` interface as the n-gram `Predictor`.

- Constraint: the text before the cursor is split into the context, the whitespace after it and the partially typed
  word. The context is cut before the trailing whitespace; the whitespace plus the typed letters (" kn") constrain the
  generated text. A prefix phase allows only tokens consistent with the constraint (" k", " kn", " know"); a word
  phase extends the word with letter and contraction tokens until the model ends it. A word's probability is the sum
  over its token paths of path probability times the probability that the next token does not continue the word,
  divided by the probability of the constraint. Paths are explored with a beam (width 8); casings are merged.
- Trailing spaces: with byte-level BPE the space belongs to the next token, so text ending in a space (the usual case
  before a next-word prediction) pushes the model toward junk tokens (section 12); the constraint above avoids it.
- After the typo fine-tuning: feed `<|typed|> prefix <|word|>` and decode the intended word with the same beam, so
  completion and correction (recie -> receive, tomm -> tomorrow) use one mechanism.
- Glued words: a letter token without a space can extend a word the model would have ended ("much" + "for" ->
  "muchfor"). These get very low probabilities; continued pretraining should reduce them further, and the decoder
  filters them if they show up among the top candidates.
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
  metrics: KSR, precision, suggestions shown, acc@0–3, top3@0, latency (mean and p99). `lm/eval.py` runs it on any
  causal LM through the decoder, plus per-word perplexity and tokens per word. For speed it defaults to 2,000 words
  per source; compare against the n-gram model at the same `--limit`.
- Resource use: file size, peak memory and per-suggestion latency on a laptop CPU with llama.cpp (milestone 3).
- Baselines on the same benchmark: the strongest n-gram model the corpus allows (no pruning, 23.35 MB; built with
  `uv run training/build.py model --out data/ngram-unpruned.bin --budget 30e6`), the bundled 5 MB n-gram model,
  the base model before fine-tuning, Supra-50M, Pythia-14M and Pythia-31M (references only).
- Robustness: the benchmark with synthetic typos in the typed prefixes, before and after typo fine-tuning.
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
1. Continued pretraining on the existing corpus, starting with a short pilot that checks how fast the domain gap
   closes.
   Before milestone 2: a CPU latency check (the decoder in PyTorch on the CPU as an upper bound, then `llama-bench` on
   an F16 / Q8_0 GGUF conversion), because a miss would change the decoder and possibly the training.
2. Typo-robust fine-tuning.
3. GGUF export (Q8_0, Q4_K_M), evaluation against the baselines, CPU latency and memory.
4. Release: Hugging Face repository `Typeling-25M` with fp16 weights, the GGUF files, model card, NOTICE and the
   reference decoder.
5. Optional: more data (section 3), if milestone 3 shows the corpus is the bottleneck.
6. Optional: integration into the n-gram kernel as an optional dependency, in two tiers like the Lexo keyboard: the
   n-gram model and the user model answer instantly, and the language model's ranking is fused in when it arrives.

Fallbacks if the fine-tuned base model disappoints: the same pipeline on Supra-50M-Base; failing that, train a
15M-parameter model from scratch with a 16k BPE tokenizer on an expanded corpus (an earlier version of this plan,
kept in git history).

## 10. Risks

- **Base model maturity.** Supra2-Medium-Base comes from a small lab, is labeled an efficiency experiment, has few
  downloads and unverified benchmark claims; milestone 0 checked it on our benchmark.
- **Capacity.** It is 1–2 KSR points behind Supra-50M zero-shot; continued pretraining has to open a 3-point lead over
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
  reproduction of training text before release.

## 11. Repository layout

```text
lm/
  decode.py      typing decoder (beam over words, typed-prefix completion)          implemented
  eval.py        typing benchmark, perplexity and simulated Q4 for causal LMs      implemented
  data.py        tokenize data/corpus into packed token streams                    implemented
  typos.py       noise model for typo-robust fine-tuning
  train.py       continued pretraining (fine-tuning to follow) (PyTorch)            implemented
  export.py      safetensors, GGUF, model card
```

Training dependencies (torch, transformers, gguf) are declared as uv inline script metadata, never as runtime
dependencies of the kernel package.

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
- With 2,000 words per source, differences of about a point are within sampling noise.

**CPU latency, upper bound.** The full-run model through the real decoder in PyTorch (float32, eager) on the laptop
CPU with 4 threads, 300 words per source (`lm/eval.py --device cpu --threads 4 --no-ppl --limit 300`), per query
mean (p95 / p99): SODA 21.2 ms (33.9 / 41.8), WildChat 18.4 ms (30.9 / 39.3), IRC 18.4 ms (34.0 / 46.8), FineWeb
27.5 ms (53.3 / 76.9). Short texts meet the target (p95 ≤ 50 ms, p99 ≤ 100 ms) with room to spare, before any
llama.cpp speedup. Long documents are slower because the decoder cuts the context to the last 2,000 characters on
every word, which shifts the token prefix and defeats the key/value cache once a text is longer than that; moving the
cut point in large steps instead would fix it.
