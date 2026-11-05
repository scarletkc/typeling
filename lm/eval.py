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
Evaluates a causal subword language model, through the typing decoder (lm/decode.py), on bench/eval.py's typing
simulation, so the results are directly comparable with the n-gram model's.

    uv run lm/eval.py --model SupraLabs/Supra-50M-Base               # typing benchmark + perplexity, dev split
    uv run lm/eval.py --model EleutherAI/pythia-31m --ppl-only       # perplexity and tokens per word only
    uv run lm/eval.py --model SupraLabs/Supra-50M-Base --probe       # example suggestions and tokenization

Besides bench/eval.py's metrics (see its docstring), per source:
    PPL/word   exp(negative log-likelihood of all tokens of the text / number of whitespace-separated words), over the
               whole dev split; comparable across tokenizers
    Tok/word   tokens per whitespace-separated word
The default --limit is lower than bench/eval.py's because every keystroke costs several model calls; run the n-gram
model with the same --limit when comparing.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.eval import DOMAINS, GRID, RESULTS, load, score, simulate  # noqa: E402
from decode import LMPredictor  # noqa: E402

PROBES = ("Let me know ", "Thank you so ", "I'm going to ", "See you ", "Can you send me the ", "lol that's so ",
          "idk what ", "Let me kn", "Thank you so mu", "I don't kn", "I will rece", "Have a great we")
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


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Typing benchmark for causal subword language models")
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local directory")
    ap.add_argument("--sources", nargs="+", default=[s for d in DOMAINS.values() for s in d],
                    help="corpora to evaluate (default: soda wildchat oasst2 irc fineweb)")
    ap.add_argument("--limit", type=int, default=2000, help="max target words per source (0 = no limit)")
    ap.add_argument("--beam", type=int, default=8, help="beam width of the decoder")
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
    ap.add_argument("--final", action="store_true", help="use the sealed test split")
    ap.add_argument("--note", default="", help="note stored in results.jsonl")
    ap.add_argument("--no-save", action="store_true", help="do not append to bench/results.jsonl")
    args = ap.parse_args()

    split = "test" if args.final else "dev"
    start = args.start.encode().decode("unicode_escape") if args.start is not None else None
    if args.threads:
        torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    p = LMPredictor(args.model, device=args.device, dtype=getattr(torch, args.dtype), start=start, beam=args.beam)
    load_ms = (time.perf_counter() - t0) * 1000
    if args.simulate_q4:
        simulate_q4(p.model)
    device = p.device + (f" ({torch.get_num_threads()} threads)" if p.device == "cpu" else "")
    print(f"Model {args.model} ({p.params / 1e6:.1f}M parameters, {args.dtype}"
          + (", simulated Q4" if args.simulate_q4 else "") + f", {device}, loaded in "
          f"{load_ms:.0f} ms), start marker {p.start!r}, beam {p.beam}, {split} split, "
          f"thresholds: next word {p.next_threshold} / completion {p.complete_threshold}")
    if args.probe:
        probe(p)
        return

    if args.ppl_only:
        print("| Source | PPL/word | Tok/word |\n|---|---|---|")
    else:
        print("| Source | Words | KSR | Precision | Shown | KSR (always shown) | acc@0 | acc@1 | acc@2 | acc@3 | top3@0 "
              "| PPL/word | Tok/word | ms/query (p95 / p99) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled = {d: ([], []) for d in DOMAINS}
    rows = []
    for src in args.sources:
        texts = load(src, split)
        ppl, tpw = (math.nan, math.nan) if args.no_ppl and not args.ppl_only else perplexity(p, texts)
        if args.ppl_only:
            print(f"| {src} | {ppl:.1f} | {tpw:.2f} |", flush=True)
            rows.append({"source": src, "ppl_word": ppl, "tokens_per_word": tpw})
            continue
        words, events, st = simulate(p, texts, args.limit, False)
        d = next(k for k, v in DOMAINS.items() if src in v)
        pooled[d][0].extend(words)
        pooled[d][1].extend(events)
        s, s0 = score(words, events, p.next_threshold, p.complete_threshold), score(words, events, 0, 0)
        rows.append({"source": src, **st, **s, "ksr_always": s0["ksr"], "ppl_word": ppl, "tokens_per_word": tpw})
        print(f"| {src} | {st['words']:,} | {s['ksr']:.2%} | {s['precision']:.1%} | {s['shown']:.2f} | "
              f"{s0['ksr']:.2%} | {st['acc@0']:.2%} | {st['acc@1']:.2%} | {st['acc@2']:.2%} | {st['acc@3']:.2%} | "
              f"{st['top3@0']:.2%} | {ppl:.1f} | {tpw:.2f} | {st['us_mean'] / 1000:.1f} "
              f"({st['us_p95'] / 1000:.1f} / {st['us_p99'] / 1000:.1f}) |", flush=True)
    for d, (words, events) in pooled.items():
        if words:
            s, s0 = score(words, events, p.next_threshold, p.complete_threshold), score(words, events, 0, 0)
            print(f"| **{d}** | {len(words):,} | **{s['ksr']:.2%}** | **{s['precision']:.1%}** | {s['shown']:.2f} | "
                  f"{s0['ksr']:.2%} | | | | | | | | |")
            rows.append({"source": d, "words": len(words), **s, "ksr_always": s0["ksr"]})
            if args.grid:
                print(f"\n{d}: KSR / precision (rows: next-word threshold, columns: completion threshold)")
                print("| | " + " | ".join(f"{t:g}" for t in GRID) + " |")
                print("|---" * (len(GRID) + 1) + "|")
                for tn in GRID:
                    cells = [score(words, events, tn, tc) for tc in GRID]
                    print(f"| {tn:g} | " + " | ".join(f"{c['ksr']:.1%} / {c['precision']:.0%}" for c in cells) + " |")
                print()
    if not args.no_save:
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "split": split,
                                "model": args.model, "params": p.params, "dtype": args.dtype, "device": device,
                                "simulate_q4": args.simulate_q4, "start": p.start,
                                "beam": p.beam, "t_next": p.next_threshold, "t_complete": p.complete_threshold,
                                "limit": args.limit, "ppl_only": args.ppl_only, "load_ms": load_ms,
                                "note": args.note, "rows": rows}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
