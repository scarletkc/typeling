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
