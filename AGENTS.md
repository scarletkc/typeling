# AGENTS.md

Typeling: offline English typing prediction. The n-gram autocomplete kernel (`typeling/`, Python standard library
only), its model build script (`training/build.py`) and benchmark (`bench/eval.py`); Typeling-25M, a small
transformer model (training and evaluation in `lm/`, plan and results in `docs/model-plan.md`), and its typing
decoder `typeling/lm.py` (optional extra `typeling[lm]`). See README.md for an overview.

## Commands

```bash
python -m unittest discover tests            # unit tests (use the bundled typeling/en.bin)
python bench/eval.py --no-save               # dev-split benchmark (chat + writing); --grid for a threshold grid
python -m typeling --bench                   # per-keystroke latency
python -m typeling                           # interactive demo (also works in the Windows terminal)

uv run training/build.py fetch               # download and clean corpora into data/corpus/ (a few hundred MB; finished sources are skipped)
uv run training/build.py model               # count, prune to 5 MB and export typeling/en.bin (about 30 s)
uv run training/build.py model --out data/x.bin --budget 3e6   # build to another path/size without overwriting en.bin
python bench/eval.py --model data/x.bin --no-save              # evaluate that alternate model

# Typeling-25M (docs/model-plan.md); PyTorch with CUDA 12.8 in an isolated uv environment
uv run lm/data.py --model SupraLabs/Supra2-Medium-Base                # tokenize data/corpus into data/lm/tokens/ (about 1 min)
uv run lm/data.py --model SupraLabs/Supra2-Medium-Base --space-split 0.01   # with the trailing-space augmentation (+space0.01)
uv run lm/train.py --model SupraLabs/Supra2-Medium-Base --tokens 20e6 --out data/lm/runs/pilot   # continued pretraining (about 60k tokens/s)
uv run lm/eval.py --model data/lm/runs/pilot                          # typing benchmark of a trained checkpoint (start marker: </s>)
uv run lm/eval.py --model data/lm/runs/pilot --device cpu --threads 4 --no-ppl --limit 300 --no-save   # CPU latency (p95 / p99)
uv run lm/eval.py --model SupraLabs/Supra2-Medium-Base --start '\n'   # zero-shot base model (never saw </s>; use a newline)
uv run lm/eval.py --model SupraLabs/Supra2-Medium-Base --ppl-only     # perplexity and tokens per word only
uv run lm/eval.py --model SupraLabs/Supra2-Medium-Base --probe        # example suggestions and tokenizations
uv run lm/eval.py --model SupraLabs/Supra-50M-Base --simulate-q4      # weights rounded as GGUF Q4_0 (pessimistic Q4_K_M)
python bench/eval.py --limit 2000 --no-save                           # the n-gram model at the same limit, for comparison
uv run lm/eval.py --model data/lm/runs/cpt-150m --typos --limit 1000  # typo benchmark (KSR clean vs every word mistyped)
uv run lm/eval.py --model data/lm/runs/cpt-150m --typo-weight 0 --ngram-weight 0   # decoder without typo repair / n-gram fusion
uv run lm/eval.py --ngram --typos --limit 1000                        # typo benchmark of the n-gram model
TYPELING_LM=data/release/Typeling-25M uv run tests/test_lm.py         # decoder tests (skipped by unittest discover without it)
uv run lm/export.py --model data/lm/runs/cpt-150m --llama-cpp data/tools/llama.cpp   # release files in data/release/ (fp16 + Q8_0 GGUF + cards)
./data/tools/llama-bin/llama-bench.exe -m data/release/Typeling-25M-GGUF/Typeling-25M-Q8_0.gguf -t 4   # llama.cpp b11953 (data/tools/, not tracked)
cd paper && ../data/tools/tectonic/tectonic.exe typeling.tex   # technical report (ACL style) -> paper/typeling.pdf (not tracked)
uv run --no-project --python 3.13 --with "transformers>=5" --with "torch>=2.7" --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match python -m typeling --lm data/lm/runs/cpt-150m --device cpu --bench   # demo with the LM
```

## Conventions

- The runtime package must stay dependency-free; numpy, pyarrow and huggingface_hub are used only by
  `training/build.py` (uv inline script metadata). The one exception is `typeling/lm.py` (PyTorch, Transformers via
  the extra `typeling[lm]`); nothing else in the package may import it.
- Release names: Hugging Face `ScarletKc/Typeling-25M` (fp16 safetensors, the default of `LMPredictor()` and
  `--lm`) and `ScarletKc/Typeling-25M-GGUF` (Q8_0); PyPI `typeling`. Publishing (Hugging Face uploads, making the
  GitHub repository public, PyPI) only on the user's explicit go-ahead.
- PyPI releases: bump `version` in `pyproject.toml`, push, then publish a GitHub release tagged `v<version>`;
  `.github/workflows/publish.yml` tests, builds and uploads with trusted publishing (environment `pypi`). A version
  number cannot be uploaded twice. Attach the technical report as `typeling-technical-report.pdf` (README.md and the
  model card link to the v0.1.1 asset). Hugging Face repo ids are case-sensitive in the model tree: use `ScarletKc/...`.
- The technical report (`paper/`) is in English and uses only numbers recorded in `docs/model-plan.md` and
  `bench/results.jsonl`; verify every new citation before adding it to `paper/references.bib`.
- When launching several background runs from PowerShell, do not pass arguments containing spaces through
  `Start-Process -ArgumentList` (it splits them); start each run as its own command.
- Tokenizer rules live only in `typeling/text.py` and are shared by training and runtime; changing them requires
  rebuilding `en.bin`.
- Any change to the model file format must update both the reader and the writer in `typeling/model.py` and bump
  `VERSION`.
- Use only permissively licensed corpora (ODC-By, CC BY, Apache-2.0); update NOTICE (and the corpus table in README.md)
  when adding a source.
- The test split (`data/corpus/*.test.jsonl`) is sealed: tune on the dev split only and run `bench/eval.py --final`
  only for final comparisons.
- Write docs, comments and program output in English.
- `lm/` scripts declare torch/transformers as uv inline script metadata (CUDA 12.8 wheel index); keep the index block
  identical across scripts so they share one environment.
- For one-off Python checks outside the scripts, use `uv run --no-project --with ... python -c ...`; without
  `--no-project`, uv creates a `.venv` in the repository and installs the package into it.
- The Supra2-Medium tokenizer adds a space before every text (`add_prefix_space`), so "\n" alone encodes as two tokens
  and the first word of a text is " word"; look up the newline token as "Ċ" (`convert_tokens_to_ids("\u010a")`).
- Git Bash rewrites arguments that look like paths (`--start "</s>"` becomes a Windows path); pass such arguments from
  PowerShell or set `MSYS_NO_PATHCONV=1`.
