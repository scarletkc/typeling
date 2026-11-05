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
Typing-error simulation for lm/eval.py's typo benchmark, and training data for typo-robust fine-tuning (an
experiment, see docs/model-plan.md).

    uv run lm/typos.py --model data/lm/runs/cpt-150m --out data/lm/tokens/typo --examples 600000

Errors (typo()): one per string, two with probability 0.2, each one of: a dropped apostrophe (dont, im), lowercase
instead of capitals (i, english), a QWERTY adjacent-key substitution, an omission, an inserted adjacent key or doubled
letter, a transposition of two neighboring letters. The keyboard layout is typeling.keyboard's, whose corrections()
the decoder's typo repair uses to undo such errors.

Training examples follow each other in one token stream that starts with </s>:
    context <|typed|> " typed letters" <|word|> " intended word" </s>
where the closing </s> also opens the next example. The context is the text before a random word of a corpus text
(at most CONTEXT_CHARS characters, 5% of its words with errors); the typed letters are its first k letters, 1 <= k <
length, and in half of the examples they get errors (then k may also be the whole word). Only the intended word and
the closing </s> are trained: a parallel uint8 stream (*.mask.bin) marks those tokens.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from typeling.keyboard import NEIGHBORS  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
TYPO_TOKENS = ("<|typed|>", "<|word|>")
DEFAULT_MIX = "fineweb=0.25,soda=0.25,wildchat=0.3,irc=0.15,oasst2=0.05"
CONTEXT_CHARS = 400
CONTEXT_ERRORS = 0.05
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")      # typeling.text.WORD
PARTIAL = re.compile(r"[A-Za-z]+(?:'[A-Za-z]*)*")      # what typeling.text.split_input accepts as a partial word


def one_error(s, rng):
    """s with one typing error, or s unchanged if the drawn error does not apply."""
    if "'" in s and rng.random() < 0.5:
        i = s.index("'")
        return s[:i] + s[i + 1:]
    if s != s.lower() and rng.random() < 0.3:
        return s.lower()
    i = int(rng.integers(len(s)))
    c = s[i]
    nb = NEIGHBORS.get(c.lower(), "")
    r = rng.random()
    if r < 0.4:
        if nb:
            d = nb[int(rng.integers(len(nb)))]
            return s[:i] + (d.upper() if c.isupper() else d) + s[i + 1:]
    elif r < 0.6:
        if len(s) > 1:
            return s[:i] + s[i + 1:]
    elif r < 0.8:
        d = nb[int(rng.integers(len(nb)))] if nb and rng.random() < 0.7 else c.lower()
        j = i + int(rng.integers(2))
        return s[:j] + (d.upper() if c.isupper() else d) + s[j:]
    elif len(s) > 1:
        i = min(i, len(s) - 2)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


def typo(s, rng):
    """s with one typing error (two with probability 0.2); always a valid partial word, at worst s itself."""
    for _ in range(1 if rng.random() < 0.8 else 2):
        for _ in range(5):
            t = one_error(s, rng)
            if t != s and PARTIAL.fullmatch(t):
                s = t
                break
    return s


def misspell(w, rng):
    """w with typing errors that change more than its case, or w itself if none applies."""
    for _ in range(5):
        t = typo(w, rng)
        if t.lower() != w.lower():
            return t
    return w


def noisy_context(text, rng, rate=CONTEXT_ERRORS):
    return WORD_RE.sub(lambda m: typo(m.group(), rng) if rng.random() < rate else m.group(), text)


def make_example(text, rng):
    """A random word of text -> (context, typed letters, word), or None."""
    words = [m for m in WORD_RE.finditer(text) if len(m.group()) > 1]
    if not words:
        return None
    m = words[int(rng.integers(len(words)))]
    w = m.group()
    noisy = rng.random() < 0.5
    k = int(rng.integers(1, len(w) + noisy))
    typed = typo(w[:k], rng) if noisy else w[:k]
    context = text[:m.start()].rstrip()
    if len(context) > CONTEXT_CHARS:
        context = context[-CONTEXT_CHARS:]
        context = context[context.find(" ") + 1:] if " " in context else context
    return noisy_context(context, rng), typed, w


def add_typo_tokens(tok):
    """Add <|typed|> and <|word|> to the tokenizer if missing; -> their ids."""
    missing = [t for t in TYPO_TOKENS if t not in tok.get_vocab()]
    if missing:
        tok.add_tokens(missing, special_tokens=True)
    return [tok.convert_tokens_to_ids(t) for t in TYPO_TOKENS]


def word_encoder(tok):
    """-> function: list of words -> token ids of " word" each (with the space that starts a word mid-text)."""
    prefix_space = tok.convert_ids_to_tokens(tok("a", add_special_tokens=False)["input_ids"][0]).startswith("\u0120")

    def encode(words):
        return tok(words if prefix_space else [" " + w for w in words], add_special_tokens=False)["input_ids"]

    return encode


def build_stream(tok, examples, batch=8192):
    """[(context, typed, word)] -> (uint16 tokens, uint8 mask)."""
    typed_id, word_id = add_typo_tokens(tok)
    eos = tok.eos_token_id
    encode_words = word_encoder(tok)
    tokens, mask = [np.array([eos], dtype=np.uint16)], [np.zeros(1, dtype=np.uint8)]
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        contexts = tok([c for c, _, _ in chunk], add_special_tokens=False)["input_ids"]
        typed = encode_words([t for _, t, _ in chunk])
        words = encode_words([w for _, _, w in chunk])
        for c, (context, _, _), t, w in zip(contexts, chunk, typed, words):
            c = c if context else []
            ids = [*c, typed_id, *t, word_id, *w, eos]
            tokens.append(np.array(ids, dtype=np.uint16))
            mask.append(np.concatenate([np.zeros(len(ids) - len(w) - 1, dtype=np.uint8),
                                        np.ones(len(w) + 1, dtype=np.uint8)]))
    return np.concatenate(tokens), np.concatenate(mask)


def main():
    from transformers import AutoTokenizer

    from data import SOURCES

    ap = argparse.ArgumentParser(description="Build the typo-robust fine-tuning streams")
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local directory (for its tokenizer)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--examples", type=int, default=600_000, help="training examples (default 600000)")
    ap.add_argument("--dev-examples", type=int, default=5000)
    ap.add_argument("--mix", default=DEFAULT_MIX, help=f"source weights (default {DEFAULT_MIX})")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mix = {k: float(v) for k, v in (part.split("=") for part in args.mix.split(","))}
    total = sum(mix.values())
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 1 << 30
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, n in (("train", args.examples), ("dev", args.dev_examples)):
        t0 = time.perf_counter()
        examples = []
        for src in SOURCES:
            if not mix.get(src):
                continue
            with open(CORPUS / f"{src}.{split}.jsonl", encoding="utf-8") as f:
                texts = [json.loads(line)["text"] for line in f]
            want = round(n * mix[src] / total)
            got = 0
            while got < want:
                ex = make_example(texts[int(rng.integers(len(texts)))], rng)
                if ex:
                    examples.append(ex)
                    got += 1
        rng.shuffle(examples)
        tokens, mask = build_stream(tok, examples)
        tokens.tofile(args.out / f"typo.{split}.bin")
        mask.tofile(args.out / f"typo.{split}.mask.bin")
        noisy = sum(t != w[:len(t)] for _, t, w in examples)
        stats[split] = {"examples": len(examples), "tokens": int(tokens.size), "trained_tokens": int(mask.sum()),
                        "with_errors": noisy}
        print(f"typo.{split}: {len(examples):,} examples ({noisy:,} with errors), {tokens.size:,} tokens, "
              f"{int(mask.sum()):,} trained, {time.perf_counter() - t0:.0f} s", flush=True)
    for context, typed, word in examples[:5]:
        print(f"  ...{context[-50:]!r} | typed {typed!r} -> {word!r}")
    with open(args.out / "typo.json", "w", encoding="utf-8") as f:
        json.dump({"model": str(args.model), "tokens": TYPO_TOKENS, "ids": add_typo_tokens(tok), "mix": mix,
                   "streams": stats}, f, indent=1)


if __name__ == "__main__":
    main()
