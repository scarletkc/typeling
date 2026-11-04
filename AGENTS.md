# AGENTS.md

Offline English autocomplete kernel (`word_predictor/`, Python standard library only), plus the model build script
(`training/build.py`) and the benchmark (`bench/eval.py`). See README.md for an overview.

## Commands

```bash
python -m unittest discover tests            # unit tests (use the bundled word_predictor/en.bin)
python bench/eval.py --no-save               # dev-split benchmark (chat + writing); --grid for a threshold grid
python -m word_predictor --bench             # per-keystroke latency
python -m word_predictor                     # interactive demo (also works in the Windows terminal)

uv run training/build.py fetch               # download and clean corpora into data/corpus/ (a few hundred MB; finished sources are skipped)
uv run training/build.py model               # count, prune to 5 MB and export word_predictor/en.bin (about 30 s)
uv run training/build.py model --out data/x.bin --budget 3e6   # build to another path/size without overwriting en.bin
python bench/eval.py --model data/x.bin --no-save              # evaluate that alternate model

# Transformer model (docs/model-plan.md); PyTorch with CUDA 12.8 in an isolated uv environment
uv run lm/eval.py --model SupraLabs/Supra2-Medium-Base --start '\n'   # typing benchmark + perplexity (2000 words/source)
uv run lm/eval.py --model SupraLabs/Supra2-Medium-Base --ppl-only     # perplexity and tokens per word only
uv run lm/eval.py --model SupraLabs/Supra2-Medium-Base --probe        # example suggestions and tokenizations
uv run lm/eval.py --model SupraLabs/Supra-50M-Base --simulate-q4      # weights rounded as GGUF Q4_0 (pessimistic Q4_K_M)
python bench/eval.py --limit 2000 --no-save                           # the n-gram model at the same limit, for comparison
```

## Conventions

- The runtime package must stay dependency-free; numpy, pyarrow and huggingface_hub are used only by
  `training/build.py` (uv inline script metadata).
- Tokenizer rules live only in `word_predictor/text.py` and are shared by training and runtime; changing them requires
  rebuilding `en.bin`.
- Any change to the model file format must update both the reader and the writer in `word_predictor/model.py` and bump
  `VERSION`.
- Use only permissively licensed corpora (ODC-By, CC BY, Apache-2.0); update NOTICE (and the corpus table in README.md)
  when adding a source.
- The test split (`data/corpus/*.test.jsonl`) is sealed: tune on the dev split only and run `bench/eval.py --final`
  only for final comparisons.
- Write docs, comments and program output in English.
- `lm/` scripts declare torch/transformers as uv inline script metadata (CUDA 12.8 wheel index); keep the index block
  identical across scripts so they share one environment.
- Git Bash rewrites arguments that look like paths (`--start "</s>"` becomes a Windows path); pass such arguments from
  PowerShell or set `MSYS_NO_PATHCONV=1`.
