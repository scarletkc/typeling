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
Evaluates a causal subword language model, through the typing decoder (typeling.lm; lm/decode.py adds the
experimental typed mode), on bench/eval.py's typing simulation, so the results are directly comparable with the
n-gram model's.

    uv run lm/eval.py --model SupraLabs/Supra-50M-Base               # typing benchmark + perplexity, dev split
    uv run lm/eval.py --model EleutherAI/pythia-31m --ppl-only       # perplexity and tokens per word only
    uv run lm/eval.py --model SupraLabs/Supra-50M-Base --probe       # example suggestions and tokenization
    uv run lm/eval.py --model data/lm/runs/cpt-150m --typos          # typo benchmark
    uv run lm/eval.py --model data/lm/runs/cpt-150m --typo-weight 0  # ... without the decoder's typo repair
    uv run lm/eval.py --ngram --typos                                # typo benchmark for the bundled n-gram model

Besides bench/eval.py's metrics (see its docstring), per source:
    PPL/word   exp(negative log-likelihood of all tokens of the text / number of whitespace-separated words), over the
               whole dev split; comparable across tokenizers
    Tok/word   tokens per whitespace-separated word
The default --limit is lower than bench/eval.py's because every keystroke costs several model calls; run the n-gram
model with the same --limit when comparing.
Typo benchmark (--typos): the typing simulation is run twice on the same words, once as is ("clean") and once by a
typist who types every word of 3+ letters with errors (lm/typos.py misspell(): one error, two with probability 0.2)
and does not notice them; accepting the intended word corrects the error. Shorter words are typed correctly. Errors
depend only on the seed and the texts, so every model gets the same ones. Per source:
    KSR clean / KSR typos   ghost-text KSR with the default thresholds, typed correctly / with errors
    Drop                    KSR clean - KSR typos
    Correct                 share of mistyped words that still end as the intended word (a suggestion of it was
                            accepted, before or after the error, or after the whole misspelled word)
    KSR@3, Correct@3        the same with the candidate bar (the intended word among the top 3)
Errors in the context are not simulated (earlier words are as in the text). At a typo rate r below 1, the KSR is
about (1 - r) x KSR clean + r x KSR typos.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.eval import (BAR, DOMAINS, GRID, HEAD, RESULTS, cells, load, rank_of, score, score_bar,  # noqa: E402
                        simulate, summarize)
from decode import TypedPredictor  # noqa: E402
from typeling import Predictor  # noqa: E402
from typeling.text import WORD_RE  # noqa: E402
from typos import misspell  # noqa: E402

PROBES = ("Let me know ", "Thank you so ", "I'm going to ", "See you ", "Can you send me the ", "lol that's so ",
          "idk what ", "Let me kn", "Thank you so mu", "I don't kn", "I will rece", "Have a great we",
          "I will recie", "See you tomm", "Let me kbow", "Thank you so mich", "I dont", "What do you thnik",
          "I looked it up in the encyc", "Sorry, I don", "Hel")
TOKEN_PROBES = ("i", "I'm", "don't", "lol", "idk", "tbh", "gonna", "u", "pls", "ok", "yeah", "tomorrow")


@torch.no_grad()
def simulate_q4(model):
    """Round the weights in place as a GGUF Q4_0 file stores them, a pessimistic stand-in for Q4_K_M: the linear
    layers in blocks of 32 weights sharing one scale, with values -8..7 times the scale (the scale is the largest
    magnitude / 8); the embedding matrix, which is also the output matrix when tied, in blocks of 16 with 6-bit
    values, as llama.cpp keeps the output tensor in Q6_K."""
    def rnd(w, block, bits):
        x = w.float().reshape(-1, block)
        top = 2 ** (bits - 1)
        d = x.gather(1, x.abs().argmax(1, keepdim=True)) / -top
        d[d == 0] = 1.0
        w.copy_(((x / d).round().clamp(-top, top - 1) * d).reshape(w.shape))

    for layer in model.model.layers:
        for m in layer.modules():
            if isinstance(m, torch.nn.Linear):
                rnd(m.weight, 32, 4)
    rnd(model.get_input_embeddings().weight, 16, 6)
    if not model.config.tie_word_embeddings:
        rnd(model.get_output_embeddings().weight, 16, 6)


@torch.inference_mode()
def perplexity(p, texts):
    """-> (per-word perplexity, tokens per word). Each text is scored from the start marker; long texts with a
    sliding window that keeps at least half a window of context."""
    window = p.max_context
    nll = words = toks = 0
    for text in texts:
        ids = p.encode(text)
        if not ids:
            continue
        words += len(text.split())
        toks += len(ids)
        seq = p.start_ids + ids
        pos = len(p.start_ids)  # next position to score
        while pos < len(seq):
            begin = max(0, pos - window // 2)
            end = min(len(seq), begin + window)
            x = torch.tensor([seq[begin:end]], device=p.device)
            logp = torch.log_softmax(p.model(input_ids=x, use_cache=False).logits[0].float(), -1)
            nll -= logp[pos - begin - 1:end - begin - 1].gather(1, x[0, pos - begin:, None]).sum().item()
            pos = end
    return math.exp(nll / max(1, words)), toks / max(1, words)


@torch.inference_mode()
def probe(p):
    """Decoder suggestions next to the raw next-token distribution of the untouched text, and tokenizations."""
    print("| Text | Decoder top 3 | Raw next tokens (text as typed) |\n|---|---|---|")
    for text in PROBES:
        cands = ", ".join(f"{w} {c:.2f}" for w, c in p.topk(text, 3))
        x = torch.tensor([p.start_ids + p.encode(text)], device=p.device)
        probs = torch.softmax(p.model(input_ids=x, use_cache=False).logits[0, -1].float(), -1)
        top = torch.topk(probs, 5)
        raw = ", ".join(f"{p.strs[i]!r} {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"| `{text}` | {cands} | {raw} |")
    print("\n| Word | Tokens after a space |\n|---|---|")
    for w in TOKEN_PROBES:
        print(f"| {w} | " + " ".join(repr(p.strs[i]) for i in p.encode(" " + w)) + " |")


def simulate_misspelled(p, texts, limit, seed=0):
    """bench/eval.py simulate() for a typist who types every word of 3+ letters as misspell(word), letter by letter;
    the steps run to the whole mistyped word. -> (word lengths, events, mistyped flags, seconds per query)"""
    rng = np.random.default_rng(seed)
    words, events, bad, times = [], [], [], []
    for text in texts:
        for m in WORD_RE.finditer(text):
            w = m.group()
            typed = misspell(w, rng) if len(w) >= 3 else w
            before = text[:m.start()]
            ev = []
            for i in range(len(typed) + (typed != w)):
                t0 = time.perf_counter()
                cands = p.topk(before + typed[:i], BAR)
                times.append(time.perf_counter() - t0)
                ev.append((rank_of(w.lower(), cands), cands[0][1] if cands else 0.0))
                if ev[-1][0] == 0 and ev[-1][1] >= GRID[-1]:
                    break  # accepted under every threshold (and from the bar)
            words.append(len(w))
            events.append(ev)
            bad.append(typed != w)
            if limit and len(words) >= limit:
                break
        if limit and len(words) >= limit:
            break
    return words, events, bad, times


def corrected(events, bad, t_next, t_complete, bar=False):
    """Share of mistyped words for which a suggestion of the intended word was accepted (bar: picked from the
    candidate bar)."""
    hits = [any(rank < BAR if bar else rank == 0 and conf >= (t_next if i == 0 else t_complete)
                for i, (rank, conf) in enumerate(ev)) for ev, b in zip(events, bad) if b]
    return sum(hits) / max(1, len(hits))


def typo_row(words, clean, typos, bad, tn, tc):
    return {"words": len(words), "mistyped": sum(bad) / len(bad), "ksr_clean": score(words, clean, tn, tc)["ksr"],
            **score(words, typos, tn, tc), "correct": corrected(typos, bad, tn, tc),
            "ksr_bar_clean": score_bar(words, clean), "ksr_bar": score_bar(words, typos),
            "correct_bar": corrected(typos, bad, tn, tc, bar=True)}


def typo_cells(source, r, bold=False):
    b = "**" if bold else ""
    return (f"| {b}{source}{b} | {r['words']:,} | {r['mistyped']:.1%} | {r['ksr_clean']:.2%} | {b}{r['ksr']:.2%}{b} | "
            f"{r['ksr_clean'] - r['ksr']:+.2%} | {r['precision']:.1%} | {r['correct']:.1%} | {r['ksr_bar_clean']:.2%} | "
            f"{b}{r['ksr_bar']:.2%}{b} | {r['ksr_bar_clean'] - r['ksr_bar']:+.2%} | {r['correct_bar']:.1%} |")


def typo_benchmark(args, p, name, split):
    """Typo benchmark (see the module docstring) for an LM through the decoder, or for the bundled n-gram model."""
    tn, tc = p.next_threshold, p.complete_threshold
    print(f"Typo benchmark: {name}, {split} split\n| Source | Words | Mistyped | KSR clean | KSR typos | Drop "
          "| Precision (typos) | Correct | KSR@3 clean | KSR@3 typos | Drop@3 | Correct@3 "
          "| ms/query with typos (p95 / p99) |\n" + "|---" * 13 + "|")
    pooled = {d: ([], [], [], []) for d in DOMAINS}   # words, clean events, typo events, mistyped flags
    rows = []
    for src in args.sources:
        texts = load(src, split)
        words, clean, st = simulate(p, texts, args.limit, False)
        words2, typos, bad, times = simulate_misspelled(p, texts, args.limit)
        assert words == words2
        d = next(k for k, v in DOMAINS.items() if src in v)
        for acc, part in zip(pooled[d], (words, clean, typos, bad)):
            acc.extend(part)
        times.sort()
        rows.append(row := {"source": src, **typo_row(words, clean, typos, bad, tn, tc),
                            **{k: v for k, v in st.items() if k.startswith(("acc@", "top3@"))},
                            "ms_mean": sum(times) / len(times) * 1000, "ms_p95": times[int(len(times) * 0.95)] * 1000,
                            "ms_p99": times[int(len(times) * 0.99)] * 1000})
        print(typo_cells(src, row) + f" {row['ms_mean']:.1f} ({row['ms_p95']:.1f} / {row['ms_p99']:.1f}) |",
              flush=True)
    for d, (words, clean, typos, bad) in pooled.items():
        if words:
            rows.append(row := {"source": d, **typo_row(words, clean, typos, bad, tn, tc)})
            print(typo_cells(d, row, bold=True) + " |")
    if not args.no_save:
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "split": split, "typos": True,
                                "model": "ngram" if args.ngram else args.model, "name": name, "limit": args.limit,
                                "note": args.note, "rows": rows}, ensure_ascii=False) + "\n")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Typing benchmark for causal subword language models")
    ap.add_argument("--model", help="Hugging Face repo id or local directory")
    ap.add_argument("--ngram", action="store_true", help="evaluate the bundled n-gram model instead (with --typos)")
    ap.add_argument("--typos", action="store_true", help="run the typo benchmark instead of the typing benchmark")
    ap.add_argument("--sources", nargs="+", default=[s for d in DOMAINS.values() for s in d],
                    help="corpora to evaluate (default: soda wildchat oasst2 irc fineweb)")
    ap.add_argument("--limit", type=int, default=2000, help="max target words per source (0 = no limit)")
    ap.add_argument("--beam", type=int, default=8, help="beam width of the decoder")
    ap.add_argument("--typo-weight", type=float, default=None,
                    help="decoder typo repair: error probability relative to none, 0 = off (default: the decoder's)")
    ap.add_argument("--typo-min", type=int, default=None, help="typo repair from this many typed letters")
    ap.add_argument("--typo-alts", type=int, default=None, help="corrections searched per query")
    ap.add_argument("--unknown-weight", type=float, default=None,
                    help="weight of the typed letters when they start no vocabulary word (default: the decoder's)")
    ap.add_argument("--context-chars", type=int, default=None,
                    help="characters of context before the cursor (default: the decoder's)")
    ap.add_argument("--ngram-weight", type=float, default=None,
                    help="share of the bundled n-gram model in the probabilities, 0 = off (default: the decoder's)")
    ap.add_argument("--start", default=None,
                    help=r"text marking the beginning of a text, escapes allowed (\n); default: the EOS token")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: PyTorch's default)")
    ap.add_argument("--no-ppl", action="store_true", help="skip the perplexity pass (for quick latency runs)")
    ap.add_argument("--simulate-q4", action="store_true",
                    help="round the weights as GGUF Q4_0 does (a pessimistic stand-in for Q4_K_M)")
    ap.add_argument("--ppl-only", action="store_true", help="only perplexity and tokens per word")
    ap.add_argument("--probe", action="store_true", help="only print example suggestions and tokenizations")
    ap.add_argument("--grid", action="store_true", help="print KSR / precision for a grid of thresholds")
    ap.add_argument("--learn", action="store_true",
                    help="call learn() after each text (user model, empty at the start of each source)")
    ap.add_argument("--final", action="store_true", help="use the sealed test split")
    ap.add_argument("--note", default="", help="note stored in results.jsonl")
    ap.add_argument("--no-save", action="store_true", help="do not append to bench/results.jsonl")
    args = ap.parse_args()

    if not args.model and not args.ngram:
        ap.error("--model or --ngram is required")
    if args.ngram and not args.typos:
        ap.error("--ngram only works with --typos; use bench/eval.py for the n-gram typing benchmark")
    split = "test" if args.final else "dev"
    if args.ngram:
        typo_benchmark(args, Predictor(), "n-gram (bundled en.bin)", split)
        return
    start = args.start.encode().decode("unicode_escape") if args.start is not None else None
    if args.threads:
        torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    opts = {k: v for k, v in (("typo_weight", args.typo_weight), ("typo_min", args.typo_min),
                              ("typo_alts", args.typo_alts), ("unknown_weight", args.unknown_weight),
                              ("ngram_weight", args.ngram_weight), ("context_chars", args.context_chars))
            if v is not None}
    p = TypedPredictor(args.model, device=args.device, dtype=getattr(torch, args.dtype), start=start, beam=args.beam,
                       **opts)
    load_ms = (time.perf_counter() - t0) * 1000
    if args.simulate_q4:
        simulate_q4(p.model)
    device = p.device + (f" ({torch.get_num_threads()} threads)" if p.device == "cpu" else "")
    repair = (f", typo repair (weight {p.typo_weight:g}, from {p.typo_min} letters, {p.typo_alts} corrections, "
              f"unknown {p.unknown_weight:g})" if p.typo_weight else "")
    repair += (f", n-gram weight {p.ngram_weight:g}" if p.ngram_weight else "") + f", context {p.context_chars} chars"
    repair += ", typed mode" if p.typed_ids else ""
    print(f"Model {args.model} ({p.params / 1e6:.1f}M parameters, {args.dtype}"
          + (", simulated Q4" if args.simulate_q4 else "") + f", {device}, loaded in "
          f"{load_ms:.0f} ms), start marker {p.start!r}, beam {p.beam}{repair}, {split} split, "
          f"thresholds: next word {p.next_threshold} / completion {p.complete_threshold}"
          + (", simulated learning" if args.learn else ""))
    if args.typos:
        typo_benchmark(args, p, f"{args.model}{repair}", split)
        return
    if args.probe:
        probe(p)
        return

    if args.ppl_only:
        print("| Source | PPL/word | Tok/word |\n|---|---|---|")
    else:
        print(HEAD + " | PPL/word | Tok/word | ms/query (p95 / p99) |\n" + "|---" * 14 + "|")
    pooled = {d: ([], []) for d in DOMAINS}
    rows = []
    for src in args.sources:
        texts = load(src, split)
        ppl, tpw = (math.nan, math.nan) if args.no_ppl and not args.ppl_only else perplexity(p, texts)
        if args.ppl_only:
            print(f"| {src} | {ppl:.1f} | {tpw:.2f} |", flush=True)
            rows.append({"source": src, "ppl_word": ppl, "tokens_per_word": tpw})
            continue
        p.user.__init__()   # start each source with an empty user model
        words, events, st = simulate(p, texts, args.limit, args.learn)
        d = next(k for k, v in DOMAINS.items() if src in v)
        pooled[d][0].extend(words)
        pooled[d][1].extend(events)
        s = summarize(words, events, p.next_threshold, p.complete_threshold)
        rows.append({"source": src, **st, **s, "ppl_word": ppl, "tokens_per_word": tpw})
        print(cells(src, st["words"], s, st) + f" | {ppl:.1f} | {tpw:.2f} | {st['us_mean'] / 1000:.1f} "
              f"({st['us_p95'] / 1000:.1f} / {st['us_p99'] / 1000:.1f}) |", flush=True)
    for d, (words, events) in pooled.items():
        if words:
            s = summarize(words, events, p.next_threshold, p.complete_threshold)
            print(cells(d, len(words), s) + " | | | |")
            rows.append({"source": d, "words": len(words), **s})
            if args.grid:
                print(f"\n{d}: KSR / precision (rows: next-word threshold, columns: completion threshold)")
                print("| | " + " | ".join(f"{t:g}" for t in GRID) + " |")
                print("|---" * (len(GRID) + 1) + "|")
                for tn in GRID:
                    row = [score(words, events, tn, tc) for tc in GRID]
                    print(f"| {tn:g} | " + " | ".join(f"{c['ksr']:.1%} / {c['precision']:.0%}" for c in row) + " |")
                print()
    if not args.no_save:
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "split": split,
                                "model": args.model, "params": p.params, "dtype": args.dtype, "device": device,
                                "simulate_q4": args.simulate_q4, "start": p.start, "typo_weight": p.typo_weight,
                                "typo_min": p.typo_min, "typo_alts": p.typo_alts, "unknown_weight": p.unknown_weight,
                                "ngram_weight": p.ngram_weight, "context_chars": p.context_chars,
                                "learn": args.learn,
                                "beam": p.beam, "t_next": p.next_threshold, "t_complete": p.complete_threshold,
                                "limit": args.limit, "ppl_only": args.ppl_only, "load_ms": load_ms,
                                "note": args.note, "rows": rows}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
