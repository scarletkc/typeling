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
Continued pretraining (milestone 1 of docs/model-plan.md): full-parameter training of a causal LM on the token streams
written by lm/data.py.

    uv run lm/data.py --model SupraLabs/Supra2-Medium-Base                                     # once
    uv run lm/train.py --model SupraLabs/Supra2-Medium-Base --tokens 20e6 --out data/lm/runs/pilot
    uv run lm/eval.py --model data/lm/runs/pilot                                               # typing benchmark

Before training, the end-of-text token's embedding is set to the newline token's: the base models never saw it, and it
separates the texts in the training streams. Each training sequence is a random window of --seq tokens from one
source, the source drawn with the --mix weights. Precision: float32 weights and optimizer state, bfloat16 autocast.
Schedule: linear warmup over --warmup of the steps, then cosine decay to zero. The dev loss of every source (mean
cross-entropy per token on fixed dev windows) is measured before training and every --eval-every steps; the result is
saved with save_pretrained, with log.jsonl and train.json next to it.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import SOURCES, token_dir

DEFAULT_MIX = "fineweb=0.3,soda=0.25,wildchat=0.25,irc=0.15,oasst2=0.05"


def parse_mix(text):
    mix = {k: float(v) for k, v in (part.split("=") for part in text.split(","))}
    unknown = set(mix) - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources in --mix: {', '.join(sorted(unknown))}")
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items() if v > 0}


def streams(directory, split, sources):
    return {s: np.memmap(directory / f"{s}.{split}.bin", dtype=np.uint16, mode="r") for s in sources}


def to_device(rows, device):
    return torch.from_numpy(np.stack(rows).astype(np.int64)).to(device, non_blocking=True)


def loss_of(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x[:, :-1]).logits
    return F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def evaluate(model, dev, seq, device, windows=64, batch=16):
    """Mean cross-entropy per token of each dev stream, on its first `windows` non-overlapping windows."""
    model.eval()
    out = {}
    for src, arr in dev.items():
        n = min(windows, (len(arr) - 1) // seq)
        total = 0.0
        for b in range(0, n, batch):
            x = to_device([arr[i * seq:(i + 1) * seq + 1] for i in range(b, min(n, b + batch))], device)
            total += loss_of(model, x).item() * len(x)
        out[src] = total / n
    model.train()
    return out


def fmt(losses):
    return " ".join(f"{k} {v:.3f}" for k, v in losses.items())


def main():
    ap = argparse.ArgumentParser(description="Continued pretraining of a causal LM")
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local directory")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--tokens", type=float, default=20e6, help="training tokens (default 20e6)")
    ap.add_argument("--mix", default=DEFAULT_MIX, help=f"source weights (default {DEFAULT_MIX})")
    ap.add_argument("--seq", type=int, default=512, help="sequence length (default 512)")
    ap.add_argument("--batch", type=int, default=32, help="sequences per micro-batch (default 32)")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step (default 2)")
    ap.add_argument("--lr", type=float, default=6e-4, help="peak learning rate (default 6e-4)")
    ap.add_argument("--warmup", type=float, default=0.05, help="share of steps for linear warmup (default 0.05)")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100, help="steps between dev evaluations (default 100)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.config.use_cache = False
    newline = tok.convert_tokens_to_ids("\u010a")  # byte-level BPE writes the newline byte as "Ċ"
    if newline is None or newline == tok.unk_token_id:
        raise SystemExit("no newline token; cannot initialize the end-of-text token from it")
    with torch.no_grad():
        for m in {model.get_input_embeddings(), model.get_output_embeddings()}:
            m.weight[tok.eos_token_id] = m.weight[newline]

    directory = token_dir(args.model)
    mix = parse_mix(args.mix)
    train = streams(directory, "train", mix)
    dev = streams(directory, "dev", SOURCES)
    names, probs = list(mix), np.array(list(mix.values()))
    per_step = args.batch * args.seq * args.accum
    steps = math.ceil(args.tokens / per_step)
    warmup = max(1, round(steps * args.warmup))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    def batch():
        rows = []
        for i in rng.choice(len(names), size=args.batch, p=probs):
            arr = train[names[i]]
            start = int(rng.integers(0, len(arr) - args.seq - 1))
            rows.append(arr[start:start + args.seq + 1])
        return to_device(rows, device)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "w", encoding="utf-8")
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {params / 1e6:.1f}M parameters; {steps} steps of {per_step:,} tokens "
          f"({steps * per_step / 1e6:.1f}M tokens), peak lr {args.lr:g}, mix {fmt(mix)}", flush=True)
    before = evaluate(model, dev, args.seq, device)
    print(f"dev loss before: {fmt(before)}", flush=True)
    log.write(json.dumps({"step": 0, "dev": before}) + "\n")

    model.train()
    t0 = time.perf_counter()
    dev_loss = before
    for step in range(steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        train_loss = 0.0
        for _ in range(args.accum):
            loss = loss_of(model, batch()) / args.accum
            loss.backward()
            train_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        record = {"step": done, "loss": train_loss, "lr": lr, "grad_norm": grad_norm}
        if done % args.log_every == 0 or done == steps:
            elapsed = time.perf_counter() - t0
            rate = done * per_step / elapsed
            print(f"step {done}/{steps}  loss {train_loss:.4f}  lr {lr:.2e}  grad norm {grad_norm:.2f}  "
                  f"{rate / 1000:.0f}k tokens/s  {elapsed / 60:.1f} min, about {(steps - done) * per_step / rate / 60:.1f} "
                  "min left", flush=True)
        if done % args.eval_every == 0 or done == steps:
            dev_loss = evaluate(model, dev, args.seq, device)
            record["dev"] = dev_loss
            print(f"dev loss at step {done}: {fmt(dev_loss)}", flush=True)
        log.write(json.dumps(record) + "\n")
    elapsed = time.perf_counter() - t0
    log.close()

    model.config.use_cache = True
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    summary = {"base_model": str(args.model), "tokens": steps * per_step, "steps": steps, "seconds": round(elapsed),
               "tokens_per_second": round(steps * per_step / elapsed), "dev_before": before, "dev_after": dev_loss,
               "args": {k: str(v) for k, v in vars(args).items()}}
    with open(args.out / "train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"saved to {args.out} ({elapsed / 60:.1f} min, {summary['tokens_per_second'] / 1000:.0f}k tokens/s)")


if __name__ == "__main__":
    main()
