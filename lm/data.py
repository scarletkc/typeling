# /// script
# requires-python = ">=3.13"
# dependencies = ["torch>=2.7", "transformers>=5", "huggingface_hub", "numpy"]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""
Tokenizes the corpus (data/corpus/*.{train,dev}.jsonl, built by training/build.py) with a base model's tokenizer into
flat uint16 token streams for lm/train.py. Every text (a chat turn or a document) is followed by the end-of-text token,
in corpus order, so consecutive turns of a conversation stay next to each other. The test split stays sealed.

    uv run lm/data.py --model SupraLabs/Supra2-Medium-Base    # -> data/lm/tokens/SupraLabs--Supra2-Medium-Base/
"""

import argparse
import json
import time
from itertools import chain
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
TOKENS = ROOT / "data" / "lm" / "tokens"
SOURCES = ("fineweb", "soda", "wildchat", "irc", "oasst2")
SPLITS = ("train", "dev")
BATCH = 8192


def token_dir(model):
    """Directory of the token streams for a model (a repo id or a local directory)."""
    return TOKENS / str(model).strip("/\\").replace("\\", "/").replace("/", "--")


def main():
    ap = argparse.ArgumentParser(description="Tokenize the corpus into packed token streams")
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local directory (for its tokenizer)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 1 << 30  # texts are packed later; silence the per-text length warning
    if len(tok) > 1 << 16:
        raise SystemExit(f"vocabulary of {len(tok)} tokens does not fit in uint16")
    eos = tok.eos_token_id
    out = token_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for src in SOURCES:
        for split in SPLITS:
            t0 = time.perf_counter()
            with open(CORPUS / f"{src}.{split}.jsonl", encoding="utf-8") as f:
                texts = [json.loads(line)["text"] for line in f]
            parts = []
            for i in range(0, len(texts), BATCH):
                ids = tok(texts[i:i + BATCH], add_special_tokens=False)["input_ids"]
                parts.append(np.fromiter(chain.from_iterable(x + [eos] for x in ids), dtype=np.uint16))
            arr = np.concatenate(parts)
            arr.tofile(out / f"{src}.{split}.bin")
            words = sum(len(t.split()) for t in texts)
            stats[f"{src}.{split}"] = {"texts": len(texts), "words": words, "tokens": int(arr.size)}
            print(f"{src}.{split}: {len(texts):,} texts, {words:,} words, {arr.size:,} tokens "
                  f"({(arr.size - len(texts)) / max(1, words):.2f} per word), {time.perf_counter() - t0:.0f} s",
                  flush=True)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"model": str(args.model), "eos": eos, "streams": stats}, f, indent=1)


if __name__ == "__main__":
    main()
