"""
手写权重的单词补全小模型（无训练、纯 Python、零依赖）

结构（一个线性层 + softmax）：
    输入  x : one-hot 编码，维度 = MAX_LEN × 26，第 (p, c) 位为 1 表示"第 p 个字母是 c"
    输出  y = softmax(W · x + b)，每个分量对应词表中的一个完整单词

权重全部手工设定：
    b[w]          —— 每个词的先验分数，下方 VOCAB_BIAS 里逐个手写（≈ log 词频）
    W[w, (p, c)]  —— 若单词 w 的第 p 个字母恰好是 c，则为 MATCH，否则为 MISMATCH
                     也就是说：每个词的权重行就是它自己拼写的"模板"

用法：
    python word_predictor.py th          # 预测前缀 th
    python word_predictor.py             # 交互模式
    python word_predictor.py --show the  # 查看某个词的权重行
    python word_predictor.py --dump      # 导出完整权重矩阵到 weights.csv
"""

import math
import sys

# ───────────────────────── 手写参数 ─────────────────────────

MATCH = 2.0       # 前缀第 p 个字母与该词第 p 个字母相同 → 加分
MISMATCH = -10.0  # 不同（或该词没这么长）→ 重罚

# 偏置 b：每个词一个数，越大越常用。想加词就往这里加一行。
VOCAB_BIAS = {
    # 超高频
    "the": 7.0, "of": 6.6, "and": 6.5, "to": 6.4, "in": 6.2, "is": 6.0,
    "it": 5.9, "you": 5.9, "that": 5.8, "a": 5.8, "i": 5.6, "he": 5.5,
    "was": 5.5, "for": 5.5, "on": 5.4, "are": 5.4, "with": 5.3, "as": 5.3,
    "his": 5.2, "they": 5.2, "be": 5.2, "at": 5.1, "have": 5.1, "this": 5.1,
    "from": 5.0, "or": 5.0, "had": 4.9, "by": 4.9, "not": 4.9, "but": 4.9,
    "what": 4.8,
    # 高频
    "all": 4.6, "were": 4.6, "we": 4.6, "when": 4.5, "your": 4.5, "can": 4.5,
    "said": 4.4, "there": 4.4, "use": 4.3, "an": 4.3, "which": 4.3, "she": 4.3,
    "do": 4.3, "how": 4.3, "their": 4.3, "if": 4.3, "will": 4.3, "each": 4.2,
    "up": 4.2, "about": 4.2, "out": 4.2, "so": 4.2, "other": 4.1, "then": 4.1,
    "them": 4.1, "her": 4.1, "many": 4.0, "these": 4.0, "some": 4.0,
    "would": 4.0, "make": 4.0, "like": 4.0, "time": 4.0, "him": 3.9,
    "into": 3.9, "has": 3.9, "more": 3.9, "two": 3.8, "go": 3.8, "see": 3.8,
    "no": 3.8, "my": 3.8, "look": 3.7, "could": 3.7, "people": 3.7, "who": 3.7,
    "get": 3.7, "way": 3.6, "than": 3.6, "first": 3.6, "been": 3.6, "now": 3.6,
    "know": 3.6, "just": 3.6, "did": 3.5, "new": 3.5, "write": 3.4,
    "down": 3.4, "day": 3.4, "come": 3.4, "may": 3.4, "think": 3.4,
    "good": 3.4, "also": 3.4, "only": 3.4, "number": 3.3, "find": 3.3,
    "long": 3.3, "made": 3.3, "because": 3.3, "want": 3.3, "well": 3.3,
    "here": 3.3, "water": 3.2, "call": 3.2, "thing": 3.2, "work": 3.2,
    "year": 3.2, "back": 3.2, "after": 3.2, "over": 3.2, "where": 3.2,
    "why": 3.2, "very": 3.2, "us": 3.2, "part": 3.1, "should": 3.1, "say": 3.1,
    # 中频
    "through": 3.0, "things": 3.0, "world": 3.0, "years": 3.0, "even": 3.0,
    "going": 3.0, "most": 3.0, "much": 3.0, "right": 3.0, "take": 3.0,
    "too": 3.0, "those": 3.0, "any": 3.0, "before": 3.0, "help": 3.0,
    "between": 2.9, "still": 2.9, "something": 2.9, "great": 2.9, "give": 2.9,
    "life": 2.9, "little": 2.9, "man": 2.9, "never": 2.9, "need": 2.9,
    "off": 2.9, "really": 2.9, "used": 2.9, "while": 2.9, "yes": 2.9,
    "again": 2.9, "let": 2.9, "thought": 2.8, "word": 2.8, "home": 2.8,
    "show": 2.8, "same": 2.8, "end": 2.8, "feel": 2.8, "last": 2.8,
    "must": 2.8, "old": 2.8, "own": 2.8, "tell": 2.8, "three": 2.8,
    "keep": 2.8, "kind": 2.8, "always": 2.8, "another": 2.8, "around": 2.8,
    "best": 2.8, "better": 2.8, "both": 2.8, "though": 2.7, "house": 2.7,
    "high": 2.7, "small": 2.7, "school": 2.7, "place": 2.7, "every": 2.7,
    "few": 2.7, "found": 2.7, "love": 2.7, "next": 2.7, "went": 2.7,
    "yet": 2.7, "under": 2.7, "ask": 2.7, "away": 2.7, "big": 2.7,
    "words": 2.6, "hello": 2.6, "hand": 2.6, "start": 2.6, "state": 2.6,
    "problem": 2.6, "play": 2.6, "point": 2.6, "case": 2.6, "change": 2.6,
    "course": 2.6, "different": 2.6, "four": 2.6, "fact": 2.6, "group": 2.6,
    "important": 2.6, "line": 2.6, "live": 2.6, "left": 2.6, "money": 2.6,
    "night": 2.6, "name": 2.6, "once": 2.6, "read": 2.6, "today": 2.6,
    "until": 2.6, "without": 2.6, "woman": 2.6, "book": 2.6, "system": 2.6,
    "please": 2.5, "company": 2.5, "country": 2.5, "car": 2.5, "child": 2.5,
    "city": 2.5,
    "family": 2.5, "free": 2.5, "open": 2.5, "order": 2.5, "room": 2.5,
    "run": 2.5, "turn": 2.5, "talk": 2.5, "week": 2.5, "young": 2.5,
    "able": 2.5, "already": 2.5, "business": 2.5, "nothing": 2.5,
    "children": 2.4, "hope": 2.4, "sometimes": 2.4, "study": 2.4,
    "story": 2.4, "stop": 2.4, "program": 2.4, "probably": 2.4, "power": 2.4,
    "close": 2.4, "everything": 2.4, "example": 2.4, "friend": 2.4,
    "food": 2.4, "government": 2.4, "girl": 2.4, "game": 2.4, "idea": 2.4,
    "mother": 2.4, "month": 2.4, "question": 2.4, "reason": 2.4,
    "thank": 2.4, "thanks": 2.4, "understand": 2.4, "white": 2.4,
    "whole": 2.4, "boy": 2.4, "bring": 2.4, "job": 2.4,
    # 低频
    "student": 2.3, "project": 2.3, "process": 2.3, "pretty": 2.3,
    "computer": 2.3, "class": 2.3, "door": 2.3, "easy": 2.3, "everyone": 2.3,
    "friends": 2.3, "information": 2.3, "issue": 2.3, "learn": 2.3,
    "level": 2.3, "remember": 2.3, "result": 2.3, "together": 2.3,
    "quite": 2.3, "answer": 2.3, "key": 2.3, "science": 2.2, "product": 2.2,
    "provide": 2.2, "code": 2.2, "data": 2.2, "interest": 2.2, "morning": 2.2,
    "music": 2.2, "test": 2.2, "usually": 2.2, "view": 2.2, "kid": 2.2,
    "begin": 2.2, "experience": 2.1, "language": 2.1, "learning": 2.1,
    "tomorrow": 2.1, "teacher": 2.1, "voice": 2.1, "video": 2.1,
    "yesterday": 2.1, "beautiful": 2.1, "president": 2.1,
    "community": 2.0, "complete": 2.0, "development": 2.0, "education": 2.0,
    "international": 2.0, "model": 2.0, "quick": 2.0, "welcome": 2.0,
    "user": 2.0, "evening": 1.9, "quickly": 1.9, "useful": 1.9,
    "python": 1.8, "network": 1.8, "weight": 1.8, "queen": 1.8, "quiet": 1.8,
    "zero": 1.8, "predict": 1.6, "zone": 1.6, "prediction": 1.5,
}

# ───────────────────────── 模型 ─────────────────────────

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORDS = list(VOCAB_BIAS)
MAX_LEN = max(len(w) for w in WORDS)
DIM = MAX_LEN * len(LETTERS)


def build_weights():
    """按规则手工写出 W（|V| × DIM）和 b（|V|）。"""
    W = []
    for w in WORDS:
        row = []
        for p in range(MAX_LEN):
            for c in LETTERS:
                row.append(MATCH if p < len(w) and w[p] == c else MISMATCH)
        W.append(row)
    b = [VOCAB_BIAS[w] for w in WORDS]
    return W, b


W, B = build_weights()


def encode(prefix):
    x = [0.0] * DIM
    for p, c in enumerate(prefix):
        x[p * 26 + LETTERS.index(c)] = 1.0
    return x


def forward(x):
    logits = [sum(wi * xi for wi, xi in zip(row, x)) + bi for row, bi in zip(W, B)]
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return logits, [e / s for e in exps]


def predict(prefix, k=5):
    prefix = "".join(ch for ch in prefix.lower() if ch in LETTERS)
    if not prefix:
        return prefix, [], False
    prefix = prefix[:MAX_LEN]
    logits, probs = forward(encode(prefix))
    ranked = sorted(range(len(WORDS)), key=lambda i: logits[i], reverse=True)[:k]
    # 若最高分的词也有字母对不上，说明词表里没有以该前缀开头的词
    exact = logits[ranked[0]] - B[ranked[0]] >= len(prefix) * MATCH - 1e-9
    return prefix, [(WORDS[i], probs[i], logits[i]) for i in ranked], exact


def show_row(word):
    """把某个词的权重行画成 字母×位置 的表：+ 表示 MATCH，· 表示 MISMATCH。"""
    i = WORDS.index(word)
    row = W[i]
    print(f"单词 {word!r} 的权重行（{DIM} 个数），b = {B[i]}")
    print(f"  '+' = {MATCH:+}   '·' = {MISMATCH:+}")
    print("     " + " ".join(f"{p:>2}" for p in range(MAX_LEN)))
    for ci, c in enumerate(LETTERS):
        cells = ["+" if row[p * 26 + ci] == MATCH else "·" for p in range(MAX_LEN)]
        print(f"  {c}  " + " ".join(f"{s:>2}" for s in cells))


def dump(path="weights.csv"):
    header = ["word", "bias"] + [f"p{p}_{c}" for p in range(MAX_LEN) for c in LETTERS]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for w, row, b in zip(WORDS, W, B):
            f.write(",".join([w, f"{b:g}"] + [f"{v:g}" for v in row]) + "\n")
    print(f"已导出 {len(WORDS)} × {DIM} 的 W 和 {len(WORDS)} 个 b 到 {path}")


def report(text):
    prefix, results, exact = predict(text)
    if not results:
        print("请输入英文字母")
        return
    note = "" if exact else "  （词表里没有以此开头的词，给出最接近的）"
    print(f"前缀 {prefix!r} →{note}")
    for w, p, z in results:
        bar = "█" * max(1, round(p * 30)) if p >= 0.005 else ""
        print(f"  {w:<14} {p:6.1%}  logit={z:6.2f}  {bar}")


def main(argv):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"词表 {len(WORDS)} 个词，W: {len(WORDS)}×{DIM}，b: {len(WORDS)}，"
          f"共 {len(WORDS) * DIM + len(WORDS)} 个参数")
    if argv[:1] == ["--dump"]:
        dump(argv[1] if len(argv) > 1 else "weights.csv")
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
