"""
评测：在 WikiText-2 上模拟打字，统计每个模型能省多少按键、猜得准不准。

    python experiment/evaluate.py B0                  # 开发集（valid），默认
    python experiment/evaluate.py B0 --limit 20000    # 只评前 2 万个目标词（快速试跑）
    python experiment/evaluate.py H2 --errors 40      # 额外打印 40 条随机出错样例
    python experiment/evaluate.py B0 --final          # 测试集：封存，只在最终统一比较时跑

模拟方式（逐个目标词）：
    目标词 = 段落里小写后全是 a-z 的 token；标点、数字、's 等只作为上下文，不计分
    依次"打出" 0、1、2… 个字母，每一步问模型要候选；
    第一个候选恰好是目标词时，用户按 Tab 接受（算 1 次按键）
    该词花费 = min(已打字母数 + 1, 词长)；从没猜中就是词长

指标：
    KSR       按键节省率 = 1 - 实际按键 / 全部手打按键（只算目标词的字母）
    acc@i     已打 i 个字母时，第一个候选就是目标词的比例（i=0 即"预测下一个词"）
    top3@0    已打 0 个字母时，目标词在前 3 个候选里的比例
    OOV       目标词不在模型词表里的比例（这些词怎么都补不出来）
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import REGISTRY  # noqa: E402

DATA = HERE.parent / "data" / "wikitext2"
RESULTS = HERE / "results.jsonl"
WORD = re.compile(r"^[a-z]+$")
K = 3
ACC_LENS = (0, 1, 2, 3)
H2_BUDGET = 50


def load_paragraphs(split):
    """每段返回 (小写 token, 原始大小写 token)。打分只看小写；原文只给需要大小写的对照模型。"""
    path = DATA / f"{split}.txt"
    if not path.exists():
        sys.exit(f"找不到 {path}，先运行：uv run experiment/prepare_data.py")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            orig = line.split()
            out.append(([t.lower() for t in orig], orig))
    return out


def evaluate(model, paragraphs, limit=None, progress=True, errors=0):
    n_words = keys_full = keys_used = oov = 0
    hit1 = dict.fromkeys(ACC_LENS, 0)
    seen = dict.fromkeys(ACC_LENS, 0)
    hit3_0 = 0
    t_begin = t_pred = 0.0
    n_pred = 0
    pred_times = []
    rng = random.Random(0)
    samples, n_miss = [], 0
    t_start = time.perf_counter()

    for toks, orig in paragraphs:
        if hasattr(model, "begin_paragraph"):  # 例如预训练语言模型：一次算完整段
            t0 = time.perf_counter()
            model.begin_paragraph(orig)
            t_begin += time.perf_counter() - t0
        for j, w in enumerate(toks):
            if not WORD.match(w):
                continue
            t0 = time.perf_counter()
            model.begin_word(toks[:j])
            t_begin += time.perf_counter() - t0

            n_words += 1
            keys_full += len(w)
            oov += not model.knows(w)
            cost = len(w)
            accepted = False
            for i in range(len(w)):
                if accepted and i > ACC_LENS[-1]:
                    break
                t0 = time.perf_counter()
                cands = model.predict(w[:i], K)
                dt = time.perf_counter() - t0
                t_pred += dt
                n_pred += 1
                if n_pred % 50 == 0:
                    pred_times.append(dt)
                top1 = cands[0] if cands else None
                if i in hit1:
                    seen[i] += 1
                    hit1[i] += top1 == w
                    if i == 0:
                        hit3_0 += w in cands
                    if errors and i <= 2 and top1 != w:  # 蓄水池抽样出错样例
                        n_miss += 1
                        item = (" ".join(toks[max(0, j - 8):j]), w[:i], w, list(cands))
                        if len(samples) < errors:
                            samples.append(item)
                        elif (r := rng.randrange(n_miss)) < errors:
                            samples[r] = item
                if not accepted and top1 == w:
                    accepted = True
                    cost = min(len(w), i + 1)
            keys_used += cost

            if progress and n_words % 20000 == 0:
                print(f"  …{n_words:,} 词  KSR {1 - keys_used / keys_full:.2%}", file=sys.stderr)
            if limit and n_words >= limit:
                break
        if limit and n_words >= limit:
            break

    pred_times.sort()
    return {
        "words": n_words,
        "ksr": 1 - keys_used / keys_full,
        **{f"acc@{i}": hit1[i] / max(1, seen[i]) for i in ACC_LENS},
        "top3@0": hit3_0 / max(1, n_words),
        "oov": oov / max(1, n_words),
        "begin_us": t_begin / max(1, n_words) * 1e6,
        "predict_us": t_pred / max(1, n_pred) * 1e6,
        "predict_p99_us": pred_times[int(len(pred_times) * 0.99)] * 1e6 if pred_times else 0.0,
        "seconds": time.perf_counter() - t_start,
        "_samples": samples,
    }


def previous_runs(kind, split):
    if not RESULTS.exists():
        return 0
    n = 0
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        n += r.get("kind") == kind and r.get("split") == split
    return n


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="在 WikiText-2 上模拟打字评测补全模型")
    ap.add_argument("models", nargs="+", help="模型名：" + ", ".join(REGISTRY))
    ap.add_argument("--limit", type=int, help="只评前 N 个目标词")
    ap.add_argument("--errors", type=int, default=0, help="打印 N 条随机出错样例")
    ap.add_argument("--final", action="store_true", help="用封存的测试集（只在最终比较时用）")
    ap.add_argument("--no-save", action="store_true", help="不写入 results.jsonl")
    args = ap.parse_args()

    split = "test" if args.final else "valid"
    paragraphs = load_paragraphs(split)
    for name in args.models:
        if name not in REGISTRY:
            sys.exit(f"未知模型 {name}，可选：{', '.join(REGISTRY)}")
        t0 = time.perf_counter()
        model = REGISTRY[name]()
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[{name}{' ' + model.version if model.version else ''}] {model.desc}")
        if model.kind == "H2" and split == "valid":
            print(f"  H2 开发集评测第 {previous_runs('H2', 'valid') + 1} 次（预算 {H2_BUDGET}）")
        r = evaluate(model, paragraphs, args.limit, errors=args.errors)
        samples = r.pop("_samples")
        r.update(model=name, kind=model.kind, version=model.version, split=split, limit=args.limit,
                 size_bytes=model.size_bytes(), load_ms=load_ms,
                 time=datetime.now().isoformat(timespec="seconds"))
        print(f"  数据集 {split}，{r['words']:,} 个目标词，用时 {r['seconds']:.0f} s")
        print(f"  KSR {r['ksr']:.2%}   acc@0 {r['acc@0']:.2%}  acc@1 {r['acc@1']:.2%}  "
              f"acc@2 {r['acc@2']:.2%}  acc@3 {r['acc@3']:.2%}   top3@0 {r['top3@0']:.2%}")
        print(f"  OOV {r['oov']:.2%}   体积 {r['size_bytes'] / 1e6:.2f} MB   加载 {load_ms:.0f} ms   "
              f"每词开始 {r['begin_us']:.1f} µs   每键预测 {r['predict_us']:.1f} µs "
              f"(p99 {r['predict_p99_us']:.0f} µs)")
        if hasattr(model, "notes"):
            print(f"  {model.notes()}")
        if samples:
            print(f"  出错样例（随机 {len(samples)} 条，已打 0–2 个字母时首选不对）：")
            for ctx, pre, w, cands in samples:
                print(f"    …{ctx[-70:]} ⟨{pre}⟩ → 目标 {w}，模型 {cands}")
        if not args.no_save:
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
