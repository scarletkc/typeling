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
    uv run lm/data.py --model SupraLabs/Supra2-Medium-Base --space-split 0.03    # -> ...Supra2-Medium-Base+space0.03/

--space-split P is the trailing-space augmentation of docs/model-plan.md: in the training streams (never the dev
streams), each word-initial token made of a space and ASCII letters (" know") is rewritten with probability P as a lone
space token followed by the tokens of the bare word (" " + "know"), so the model also learns to continue text that
ends in a space.
"""

import argparse
import json
import re
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
WORD_PIECE = re.compile("\u0120[A-Za-z]+")  # byte-level BPE piece of a space followed by ASCII letters


def token_dir(model):
    """Directory of the token streams for a model (a repo id or a local directory)."""
    return TOKENS / str(model).strip("/\\").replace("\\", "/").replace("/", "--")


class SpaceSplitter:
    """Rewrites a share of the word-initial tokens as a lone space token followed by the tokens of the bare word."""

    def __init__(self, tok, p, seed=0):
        self.pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.word = np.array([bool(WORD_PIECE.fullmatch(s or "")) for s in self.pieces])
        self.space = tok.convert_tokens_to_ids("\u0120")
        if self.space is None or self.space == tok.unk_token_id:
            raise SystemExit("the tokenizer has no lone space token")
        self.bpe = tok.backend_tokenizer.model
        self.p, self.rng, self.splits, self.count = p, np.random.default_rng(seed), {}, 0

    def __call__(self, arr):
        pos = np.flatnonzero(self.word[arr] & (self.rng.random(arr.size) < self.p))
        self.count += len(pos)
        parts, prev = [], 0
        for i in pos.tolist():
            t = int(arr[i])
            if t not in self.splits:
                bare = [x.id for x in self.bpe.tokenize(self.pieces[t][1:])]
                self.splits[t] = np.array([self.space, *bare], dtype=np.uint16)
            parts += [arr[prev:i], self.splits[t]]
            prev = i + 1
        parts.append(arr[prev:])
        return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description="Tokenize the corpus into packed token streams")
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local directory (for its tokenizer)")
    ap.add_argument("--space-split", type=float, default=0.0,
                    help="share of word-initial training tokens rewritten as space + bare word (default 0)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: data/lm/tokens/<model>, plus +space<P> with --space-split)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 1 << 30  # texts are packed later; silence the per-text length warning
    if len(tok) > 1 << 16:
        raise SystemExit(f"vocabulary of {len(tok)} tokens does not fit in uint16")
    eos = tok.eos_token_id
    splitter = SpaceSplitter(tok, args.space_split) if args.space_split > 0 else None
    out = args.out or token_dir(args.model)
    if splitter and not args.out:
        out = out.with_name(f"{out.name}+space{args.space_split:g}")
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
            words = sum(len(t.split()) for t in texts)
            per_word = (arr.size - len(texts)) / max(1, words)
            n_split = 0
            if splitter and split == "train":
                before = splitter.count
                arr = splitter(arr)
                n_split = splitter.count - before
            arr.tofile(out / f"{src}.{split}.bin")
            stats[f"{src}.{split}"] = {"texts": len(texts), "words": words, "tokens": int(arr.size),
                                       "space_splits": n_split}
            print(f"{src}.{split}: {len(texts):,} texts, {words:,} words, {arr.size:,} tokens ({per_word:.2f} per "
                  "word before augmentation" + (f", {n_split:,} space splits" if n_split else "")
                  + f"), {time.perf_counter() - t0:.0f} s", flush=True)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"model": str(args.model), "eos": eos, "space_split": args.space_split, "streams": stats}, f,
                  indent=1)


if __name__ == "__main__":
    main()
