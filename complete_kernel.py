"""
自动补全内核：只加载导出的权重 weights.npz，不再需要 words.txt。

模型仍是  logits = W·x + b，但推理是增量的：
    x 是前缀的 one-hot，前缀多一个字母，x 就多一个 1，
    所以 W·x 只需要加上 W 的一列（|V| 次加法）；退格就把那一列减回去。
    每敲一个键 ≈ 一次列加法 + 一次 argmax，几十微秒。

用法：
    from complete_kernel import Completer
    c = Completer()
    c.suggest("I wan")   # → "t"   （灰色补全部分）
    c.topk("comp", 5)    # → [("company", 0.188), ...]
"""

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights.npz"

# 光标前最后一个"词"：前面是行首或空白（允许跟一个左引号/括号），后面全是英文字母
_TOKEN = re.compile(r"(?:^|\s)[\"'(\[{<“‘]*([A-Za-z]+)$")


class Completer:
    def __init__(self, path=DEFAULT_WEIGHTS):
        path = Path(path)
        if not path.exists():  # 第一次用：从 words.txt 构建并导出
            import word_predictor
            word_predictor.dump(str(path))
        with np.load(path) as d:
            W = d["W"]
            self.b = d["b"].astype(np.float32)
            self.words = d["words"].tolist()
            self.match = float(d["match"])
        self.V, D = W.shape
        self.max_len = D // 26
        # 转置成 (D, |V|)：每一列在内存里连续，增量加减更快
        self.Wt = np.ascontiguousarray(W.T)
        # 状态：当前前缀对应的 W·x（只含 ±MATCH/MISMATCH 的整数，float32 精确表示）
        self._prefix = ""
        self._acc = np.zeros(self.V, dtype=np.float32)
        self._logits = np.empty(self.V, dtype=np.float32)

    # ───────── 增量推理 ─────────

    def _col(self, p, ch):
        return self.Wt[p * 26 + ord(ch) - 97]

    def _advance(self, prefix):
        """把内部 W·x 从旧前缀增量更新到新前缀。"""
        old = self._prefix
        k, n = 0, min(len(old), len(prefix))
        while k < n and old[k] == prefix[k]:
            k += 1
        if (len(old) - k) + (len(prefix) - k) > len(prefix) + 1:
            self._acc.fill(0)  # 改动比重算还多，直接从零开始
            k = 0
        else:
            for p in range(k, len(old)):
                self._acc -= self._col(p, old[p])
        for p in range(k, len(prefix)):
            self._acc += self._col(p, prefix[p])
        self._prefix = prefix

    def _forward(self, prefix):
        self._advance(prefix)
        np.add(self._acc, self.b, out=self._logits)
        return self._logits

    def _ok(self, prefix):
        return 0 < len(prefix) <= self.max_len and prefix.isascii() and prefix.isalpha()

    # ───────── 对外接口 ─────────

    def best(self, prefix):
        """前缀最可能的完整单词；词表里没有以它开头的词时返回 None。"""
        prefix = prefix.lower()
        if not self._ok(prefix):
            return None
        logits = self._forward(prefix)
        i = int(logits.argmax())
        if self._acc[i] != len(prefix) * self.match:  # 有字母对不上
            return None
        return self.words[i]

    def candidates(self, prefix, k=5):
        """前缀的前 k 个完全匹配的候选词（不算概率，最快：k 次 argmax）。"""
        prefix = prefix.lower()
        if not self._ok(prefix):
            return []
        logits = self._forward(prefix)  # 内部缓冲区，下次推理会整体重算，可以就地改
        full = len(prefix) * self.match
        out = []
        for _ in range(k):
            i = int(logits.argmax())
            if self._acc[i] != full:
                break
            out.append(self.words[i])
            logits[i] = -np.inf
        return out

    def topk(self, prefix, k=5):
        """前缀的前 k 个候选及概率（只含完全匹配的词）。"""
        prefix = prefix.lower()
        if not self._ok(prefix):
            return []
        logits = self._forward(prefix)
        top = np.argpartition(-logits, k)[:k]
        top = top[np.argsort(-logits[top])]
        z = np.exp(logits.astype(np.float64) - logits[top[0]])
        probs = z / z.sum()
        full = len(prefix) * self.match
        return [(self.words[i], float(probs[i])) for i in top if self._acc[i] == full]

    def suggest(self, text):
        """给定光标前的整行文字，返回应该显示成灰色的补全部分（可能为空）。"""
        m = _TOKEN.search(text)
        if not m:
            return ""
        token = m.group(1)
        word = self.best(token)
        if not word or len(word) <= len(token):
            return ""
        tail = word[len(token):]
        return tail.upper() if len(token) > 1 and token.isupper() else tail
