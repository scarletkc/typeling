# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
单词补全小模型：输入前缀字母，预测完整英文单词。没有任何训练，权重按规则直接写出。

结构（一个线性层 + softmax）：
    输入  x : one-hot 编码，维度 = MAX_LEN × 26，第 (p, c) 位为 1 表示"第 p 个字母是 c"
    输出  y = softmax(W · x + b)，每个分量对应词表 words.txt 中的一个完整单词

权重：
    W[w, (p, c)] = MATCH     若单词 w 的第 p 个字母是 c
                 = MISMATCH  否则（包括单词没这么长）
                 —— 每个词的权重行就是它自己拼写的"模板"
    b[w]         = ln(词频) = Zipf × ln(10)，词频来自 words.txt
                 —— 所以前缀完全匹配的那些词里，softmax 概率正好等于它们的相对词频

用法：
    python word_predictor.py              # 交互模式
    python word_predictor.py th comp      # 直接预测
    python word_predictor.py --show the   # 查看某个词的权重行
    python word_predictor.py --dump       # 导出 W、b 到 weights.npz
    （不想用全局 numpy 时，把 python 换成 uv run 即可）

词表由 build_vocab.py 生成：uv run build_vocab.py [词数]
"""

import math
import sys
from pathlib import Path

import numpy as np

# ───────────────────────── 手写参数 ─────────────────────────

MATCH = 2         # 前缀第 p 个字母与该词第 p 个字母相同 → 加分
MISMATCH = -20    # 不同 → 重罚；要大于词频偏置的跨度，才能保证对不上的词排在后面

LETTERS = "abcdefghijklmnopqrstuvwxyz"
VOCAB_PATH = Path(__file__).resolve().parent / "words.txt"

# ───────────────────────── 模型 ─────────────────────────


def load_vocab(path=VOCAB_PATH):
    words, zipf = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        w, z = line.split("\t")
        words.append(w)
        zipf.append(float(z))
    return words, np.array(zipf, dtype=np.float32)


def build_weights(words, zipf):
    """按规则写出 W（|V| × MAX_LEN·26，int8）和 b（|V|）。"""
    V, L = len(words), max(map(len, words))
    codes = np.full((V, L), -1, dtype=np.int64)
    for v, w in enumerate(words):
        codes[v, : len(w)] = np.frombuffer(w.encode("ascii"), dtype=np.uint8) - ord("a")
    W = np.full((V, L, 26), MISMATCH, dtype=np.int8)
    v_idx, p_idx = np.nonzero(codes >= 0)
    W[v_idx, p_idx, codes[v_idx, p_idx]] = MATCH
    b = zipf * np.float32(math.log(10))
    return W.reshape(V, L * 26), b, L


WORDS, ZIPF = load_vocab()
W, B, MAX_LEN = build_weights(WORDS, ZIPF)
INDEX = {w: i for i, w in enumerate(WORDS)}


def forward(prefix):
    # x 是 one-hot：W·x 就等于把 x 中为 1 的那几列取出来相加（省掉乘 0 的部分）
    cols = [p * 26 + LETTERS.index(c) for p, c in enumerate(prefix)]
    logits = W[:, cols].sum(axis=1, dtype=np.float32) + B
    z = logits.astype(np.float64) - logits.max()
    probs = np.exp(z)
    probs /= probs.sum()
    return logits, probs


def predict(text, k=5):
    prefix = "".join(ch for ch in text.lower() if ch in LETTERS)[:MAX_LEN]
    if not prefix:
        return prefix, [], False
    logits, probs = forward(prefix)
    top = np.argpartition(-logits, k)[:k]
    top = top[np.argsort(-logits[top])]
    # 若最高分的词也有字母对不上，说明词表里没有以该前缀开头的词
    exact = bool(logits[top[0]] - B[top[0]] >= len(prefix) * MATCH - 1e-4)
    return prefix, [(WORDS[i], float(probs[i]), float(logits[i])) for i in top], exact


def show_row(word):
    """把某个词的权重行画成 字母×位置 的表：+ 表示 MATCH，· 表示 MISMATCH。"""
    if word not in INDEX:
        print(f"{word!r} 不在词表里")
        return
    i = INDEX[word]
    row = W[i].reshape(MAX_LEN, 26)
    shown = min(MAX_LEN, len(word) + 2)
    print(f"单词 {word!r} 的权重行（{MAX_LEN * 26} 个数），b = {B[i]:.2f}（Zipf {ZIPF[i]:.2f}）")
    print(f"  '+' = {MATCH:+}   '·' = {MISMATCH:+}   只画前 {shown} 个位置，其余位置全是 '·'")
    print("     " + " ".join(f"{p:>2}" for p in range(shown)))
    for ci, c in enumerate(LETTERS):
        cells = ["+" if row[p, ci] == MATCH else "·" for p in range(shown)]
        print(f"  {c}  " + " ".join(f"{s:>2}" for s in cells))


def dump(path="weights.npz"):
    np.savez_compressed(path, W=W, b=B, words=np.array(WORDS),
                        match=MATCH, mismatch=MISMATCH)
    print(f"已导出 W {W.shape}、b {B.shape} 和词表到 {path}")


def report(text):
    prefix, results, exact = predict(text)
    if not results:
        print("请输入英文字母")
        return
    note = "" if exact else "  （词表里没有以此开头的词，给出最接近的）"
    print(f"前缀 {prefix!r} →{note}")
    for w, p, z in results:
        bar = "█" * max(1, round(p * 30)) if p >= 0.005 else ""
        print(f"  {w:<16} {p:6.1%}  logit={z:6.2f}  {bar}")


def main(argv):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    V = len(WORDS)
    print(f"词表 {V:,} 个词，W: {V:,}×{MAX_LEN * 26}，b: {V:,}，"
          f"共 {V * MAX_LEN * 26 + V:,} 个参数")
    if argv[:1] == ["--dump"]:
        dump(argv[1] if len(argv) > 1 else "weights.npz")
    elif argv[:1] == ["--show"] and len(argv) > 1:
        show_row(argv[1].lower())
    elif argv:
        for a in argv:
            report(a)
    else:
        print("输入字母前缀（回车退出）")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                break
            if not s:
                break
            report(s)


if __name__ == "__main__":
    main(sys.argv[1:])
