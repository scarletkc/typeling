"""
Chat / writing benchmark: simulates typing held-out text and measures how many keystrokes the suggestions
save and how often the displayed suggestions are right.

    python bench/eval.py                      # dev split, all sources
    python bench/eval.py --limit 5000         # at most 5000 target words per source (quick run)
    python bench/eval.py --no-context         # ablation: typed letters and word frequency only
    python bench/eval.py --learn              # call learn() after each segment to simulate user learning
    python bench/eval.py --grid               # also print results for a grid of display thresholds
    python bench/eval.py --final              # sealed test split: run only for final comparisons

Simulation, per target word (every English word in the text, apostrophes included):
    "type" 0, 1, 2, ... letters and call topk(text before the cursor, 3) at each step.
    When the top candidate is the target and its probability passes the display threshold,
    the user presses Tab to accept it (1 keystroke).
    Cost of a word = min(letters typed + 1, word length), or the word length if never accepted.
    Comparisons are case-insensitive.

Metrics:
    KSR        keystroke savings rate = 1 - keystrokes used / keystrokes to type everything (target-word letters only)
    Precision  share of displayed suggestions that were accepted (a wrong suggestion distracts the user once)
    Shown      average number of suggestions displayed per target word
    acc@i      share of target words whose top candidate is correct after i typed letters, regardless of
               thresholds; acc@0 is next-word prediction accuracy
    top3@0     share of target words among the top 3 candidates with no letters typed
    OOV        share of target words missing from the model vocabulary
The "KSR (always shown)" column is the KSR with both thresholds at 0; latency is per topk() call (mean and p99).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from typeling import DEFAULT_MODEL, Predictor  # noqa: E402
from typeling.text import WORD_RE, normalize  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
RESULTS = Path(__file__).resolve().parent / "results.jsonl"
DOMAINS = {"chat": ("soda", "wildchat", "oasst2"), "casual": ("irc",), "writing": ("fineweb",)}
ACC_LENS = (0, 1, 2, 3)
GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def load(source, split):
    path = CORPUS / f"{source}.{split}.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found; run: uv run training/build.py fetch")
    with open(path, encoding="utf-8") as f:
        return [normalize(json.loads(line)["text"]) for line in f]


def simulate(p, texts, limit, learn, vocab=None):
    """Simulate typing word by word, recording (top candidate is correct, its probability) at every step;
    thresholds are applied later in score(). vocab: the model's word vocabulary for the OOV rate
    (None for subword models, which have no out-of-vocabulary words)."""
    words, events = [], []
    hit = dict.fromkeys(ACC_LENS, 0)
    seen = dict.fromkeys(ACC_LENS, 0)
    top3, oov, times = 0, 0, []
    for text in texts:
        for m in WORD_RE.finditer(text):
            w = m.group().lower()
            before = text[: m.start()]
            oov += vocab is not None and w not in vocab
            ev = []
            for i in range(len(w)):
                t0 = time.perf_counter()
                cands = p.topk(before + m.group()[:i], 3)
                times.append(time.perf_counter() - t0)
                ok = bool(cands) and cands[0][0].lower() == w
                conf = cands[0][1] if cands else 0.0
                if i in hit:
                    seen[i] += 1
                    hit[i] += ok
                    if i == 0:
                        top3 += any(c.lower() == w for c, _ in cands)
                ev.append((ok, conf))
                if ok and conf >= GRID[-1] and i >= ACC_LENS[-1]:
                    break  # accepted under every threshold from here on
            words.append(len(w))
            events.append(ev)
            if limit and len(words) >= limit:
                break
        if learn:
            p.learn(text)
        if limit and len(words) >= limit:
            break
    times.sort()
    stats = {
        "words": len(words),
        **{f"acc@{i}": hit[i] / max(1, seen[i]) for i in ACC_LENS},
        "top3@0": top3 / max(1, len(words)),
        "oov": oov / max(1, len(words)),
        "us_mean": sum(times) / max(1, len(times)) * 1e6,
        "us_p95": times[int(len(times) * 0.95)] * 1e6 if times else 0.0,
        "us_p99": times[int(len(times) * 0.99)] * 1e6 if times else 0.0,
    }
    return words, events, stats


def score(words, events, t_next, t_complete):
    """KSR, precision and suggestions shown per word for the given display thresholds."""
    full = used = shown = accepted = 0
    for n, ev in zip(words, events):
        full += n
        cost = n
        for i, (ok, conf) in enumerate(ev):
            if conf < (t_next if i == 0 else t_complete):
                continue
            shown += 1
            if ok:
                accepted += 1
                cost = min(n, i + 1)
                break
        used += cost
    return {"ksr": 1 - used / max(1, full), "precision": accepted / max(1, shown), "shown": shown / max(1, len(words))}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Chat / writing autocomplete benchmark")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="model file (default: the bundled en.bin)")
    ap.add_argument("--sources", nargs="+", default=[s for d in DOMAINS.values() for s in d],
                    help="corpora to evaluate (default: soda wildchat oasst2 irc fineweb)")
    ap.add_argument("--limit", type=int, default=20000, help="max target words per source (0 = no limit)")
    ap.add_argument("--no-context", action="store_true", help="ablation: ignore the context")
    ap.add_argument("--learn", action="store_true", help="call learn() after each segment")
    ap.add_argument("--cache", type=float, default=None, help="cache weight (default: the Predictor default)")
    ap.add_argument("--grid", action="store_true", help="print KSR / precision for a grid of thresholds")
    ap.add_argument("--final", action="store_true", help="use the sealed test split")
    ap.add_argument("--note", default="", help="note stored in results.jsonl")
    ap.add_argument("--no-save", action="store_true", help="do not append to bench/results.jsonl")
    args = ap.parse_args()

    split = "test" if args.final else "dev"
    t0 = time.perf_counter()
    p = Predictor(args.model, use_context=not args.no_context)
    load_ms = (time.perf_counter() - t0) * 1000
    if args.cache is not None:
        p.cache_weight = args.cache
    t_next, t_complete = p.next_threshold, p.complete_threshold
    print(f"Model {args.model.name} ({args.model.stat().st_size / 1e6:.2f} MB, loaded in {load_ms:.0f} ms), "
          f"{split} split, thresholds: next word {t_next} / completion {t_complete}"
          + (", no context" if args.no_context else "") + (", simulated learning" if args.learn else ""))
    print("| Source | Words | KSR | Precision | Shown | KSR (always shown) | acc@0 | acc@1 | acc@2 | acc@3 | top3@0 | OOV "
          "| µs/query (p99) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled = {d: ([], []) for d in DOMAINS}
    rows = []
    for src in args.sources:
        p.user.__init__()  # start each source with an empty user model
        words, events, st = simulate(p, load(src, split), args.limit, args.learn, p.lm.id)
        d = next(k for k, v in DOMAINS.items() if src in v)
        pooled[d][0].extend(words)
        pooled[d][1].extend(events)
        s, s0 = score(words, events, t_next, t_complete), score(words, events, 0, 0)
        row = {"source": src, **st, **s, "ksr_always": s0["ksr"]}
        rows.append(row)
        print(f"| {src} | {st['words']:,} | {s['ksr']:.2%} | {s['precision']:.1%} | {s['shown']:.2f} | "
              f"{s0['ksr']:.2%} | {st['acc@0']:.2%} | {st['acc@1']:.2%} | {st['acc@2']:.2%} | {st['acc@3']:.2%} | "
              f"{st['top3@0']:.2%} | {st['oov']:.2%} | {st['us_mean']:.0f} ({st['us_p99']:.0f}) |", flush=True)
    for d, (words, events) in pooled.items():
        if words:
            s, s0 = score(words, events, t_next, t_complete), score(words, events, 0, 0)
            print(f"| **{d}** | {len(words):,} | **{s['ksr']:.2%}** | **{s['precision']:.1%}** | {s['shown']:.2f} | "
                  f"{s0['ksr']:.2%} | | | | | | | |")
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
                                "model": args.model.name, "size_bytes": args.model.stat().st_size,
                                "meta": p.lm.meta, "no_context": args.no_context, "learn": args.learn,
                                "cache": p.cache_weight, "t_next": t_next, "t_complete": t_complete,
                                "limit": args.limit, "load_ms": load_ms, "note": args.note, "rows": rows},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
