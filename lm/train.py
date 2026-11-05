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
Full-parameter training of a causal LM on the token streams written by lm/data.py and lm/typos.py: continued
pretraining (milestone 1 of docs/model-plan.md) and typo-robust fine-tuning (milestone 2).

    uv run lm/data.py --model SupraLabs/Supra2-Medium-Base                                     # once
    uv run lm/train.py --model SupraLabs/Supra2-Medium-Base --tokens 20e6 --out data/lm/runs/pilot
    uv run lm/typos.py --model data/lm/runs/pilot --out data/lm/tokens/typo                    # once
    uv run lm/train.py --model data/lm/runs/pilot --typo-tokens --typo-data data/lm/tokens/typo \
        --mix "typo=0.5,fineweb=0.15,soda=0.12,wildchat=0.13,irc=0.08,oasst2=0.02" --lr 3e-4 --out data/lm/runs/typo
    uv run lm/eval.py --model data/lm/runs/typo                                                # typing benchmark

When training a base model (a directory without our train.json), the end-of-text token's embedding is first set to
the newline token's: the base models never saw it, and it separates the texts in the training streams. --typo-tokens
adds <|typed|> and <|word|> to the vocabulary. Each training sequence is a random window of --seq tokens from one
source, the source drawn with the --mix weights; windows of the typo stream start at an example boundary, and only
the tokens its mask marks are trained. The loss is the mean over sequences of each sequence's mean token loss, so a
typo window counts as much as a text window although it trains far fewer tokens. Precision: float32 weights and
optimizer state, bfloat16 autocast. Schedule: linear warmup over --warmup of the steps, then cosine decay to zero.
The dev loss of every source (mean cross-entropy per trained token on fixed dev windows) is measured before training
and every --eval-every steps; the result is saved with save_pretrained, with log.jsonl and train.json next to it.
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
from typos import add_typo_tokens

DEFAULT_MIX = "fineweb=0.3,soda=0.25,wildchat=0.25,irc=0.15,oasst2=0.05"
TYPO = "typo"


def parse_mix(text):
    mix = {k: float(v) for k, v in (part.split("=") for part in text.split(","))}
    unknown = set(mix) - set(SOURCES) - {TYPO}
    if unknown:
        raise SystemExit(f"unknown sources in --mix: {', '.join(sorted(unknown))}")
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items() if v > 0}


class Stream:
    """A token stream, with the mask of trained tokens and the example boundaries if it has a mask file."""

    def __init__(self, path, eos):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        mask = path.with_name(path.name.replace(".bin", ".mask.bin"))
        self.mask = np.memmap(mask, dtype=np.uint8, mode="r") if mask.exists() else None
        self.starts = np.flatnonzero(self.tokens == eos) if self.mask is not None else None

    def window(self, start, length):
        x = self.tokens[start:start + length]
        return x, (self.mask[start:start + length] if self.mask is not None else np.ones(len(x), dtype=np.uint8))

    def random_start(self, rng, length):
        if self.starts is None:
            return int(rng.integers(0, len(self.tokens) - length))
        return int(self.starts[rng.integers(0, np.searchsorted(self.starts, len(self.tokens) - length))])


def to_device(rows, device):
    x = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64)).to(device, non_blocking=True)
    m = torch.from_numpy(np.stack([r[1] for r in rows]).astype(np.float32)).to(device, non_blocking=True)
    return x, m


def token_losses(model, x, m):
    """-> (cross-entropy of every predicted token, its weight: 1 if trained, 0 otherwise), both (batch, seq)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x[:, :-1]).logits
    ce = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(x.shape[0], -1)
    return ce, m[:, 1:]


def loss_of(model, x, m):
    """Mean over sequences of each sequence's mean loss over its trained tokens."""
    ce, w = token_losses(model, x, m)
    n = w.sum(1)
    keep = n > 0
    return ((ce * w).sum(1)[keep] / n[keep]).mean()


@torch.no_grad()
def evaluate(model, dev, seq, device, windows=64, batch=16):
    """Mean cross-entropy per trained token of each dev stream, on its first `windows` non-overlapping windows."""
    model.eval()
    out = {}
    for src, stream in dev.items():
        n = min(windows, (len(stream.tokens) - 1) // seq)
        total = count = 0.0
        for b in range(0, n, batch):
            x, m = to_device([stream.window(i * seq, seq + 1) for i in range(b, min(n, b + batch))], device)
            ce, w = token_losses(model, x, m)
            total += (ce * w).sum().item()
            count += w.sum().item()
        out[src] = total / max(1.0, count)
    model.train()
    return out


def fmt(losses):
    return " ".join(f"{k} {v:.3f}" for k, v in losses.items())


def save(model, tok, directory):
    model.config.use_cache = True
    model.save_pretrained(directory)
    tok.save_pretrained(directory)
    model.config.use_cache = False


def main():
    ap = argparse.ArgumentParser(description="Continued pretraining and fine-tuning of a causal LM")
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local directory")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--data", type=Path, default=None,
                    help="token streams from lm/data.py (default: data/lm/tokens/<model>)")
    ap.add_argument("--typo-data", type=Path, default=None,
                    help="directory of typo.{train,dev}.bin from lm/typos.py (default: --data)")
    ap.add_argument("--typo-tokens", action="store_true", help="add <|typed|> and <|word|> to the vocabulary")
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
    ap.add_argument("--save-every", type=int, default=0, help="steps between checkpoints (default 0: only at the end)")
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
    if not (Path(args.model) / "train.json").exists():  # a base model: </s> was never trained
        newline = tok.convert_tokens_to_ids("\u010a")  # byte-level BPE writes the newline byte as "Ċ"
        if newline is None or newline == tok.unk_token_id:
            raise SystemExit("no newline token; cannot initialize the end-of-text token from it")
        with torch.no_grad():
            for m in {model.get_input_embeddings(), model.get_output_embeddings()}:
                m.weight[tok.eos_token_id] = m.weight[newline]
        print("initialized </s> from the newline token", flush=True)
    if args.typo_tokens:
        ids = add_typo_tokens(tok)
        if max(ids) >= model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
        print(f"typo tokens: {dict(zip(tok.convert_ids_to_tokens(ids), ids))}", flush=True)

    directory = args.data or token_dir(args.model)
    typo_dir = args.typo_data or directory
    mix = parse_mix(args.mix)
    path = lambda s, split: (typo_dir if s == TYPO else directory) / f"{s}.{split}.bin"  # noqa: E731
    train = {s: Stream(path(s, "train"), tok.eos_token_id) for s in mix}
    dev = {s: Stream(path(s, "dev"), tok.eos_token_id) for s in (*SOURCES, *([TYPO] if TYPO in mix else []))}
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
            stream = train[names[i]]
            rows.append(stream.window(stream.random_start(rng, args.seq + 1), args.seq + 1))
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
          f"({steps * per_step / 1e6:.1f}M tokens), peak lr {args.lr:g}, mix {fmt(mix)}, data {directory.name}",
          flush=True)
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
            loss = loss_of(model, *batch()) / args.accum
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
        log.flush()
        if args.save_every and done % args.save_every == 0 and done < steps:
            save(model, tok, args.out / "checkpoint")
            print(f"checkpoint saved at step {done}", flush=True)
    elapsed = time.perf_counter() - t0
    log.close()

    save(model, tok, args.out)
    summary = {"base_model": str(args.model), "tokens": steps * per_step, "steps": steps, "seconds": round(elapsed),
               "tokens_per_second": round(steps * per_step / elapsed), "dev_before": before, "dev_after": dev_loss,
               "args": {k: str(v) for k, v in vars(args).items()}}
    with open(args.out / "train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"saved to {args.out} ({elapsed / 60:.1f} min, {summary['tokens_per_second'] / 1000:.0f}k tokens/s)")


if __name__ == "__main__":
    main()
