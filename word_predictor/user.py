"""
User model: counts of the words and word pairs the user has typed, stored only on this machine.

    P_user(w | previous token v) = (c(v, w) + ALPHA * c(w) / N) / (c(v) + ALPHA)
Its weight in the mixture grows with the amount of user data: lambda = MAX_WEIGHT * N / (N + HALF).
"""

import heapq
import json
import os
from bisect import bisect_left, insort
from pathlib import Path

from .text import BOUNDARY, PUNCT, is_word, normalize, tokenize

ALPHA = 2.0
MAX_WEIGHT = 0.3
HALF = 2000          # learned word count at which the weight reaches half of MAX_WEIGHT
MAX_BIGRAMS = 200_000


class UserModel:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.uni, self.bi, self.ctx, self.forms = {}, {}, {}, {}
        self.total = 0
        self.sorted = []
        if self.path and self.path.exists():
            self._load()

    def learn(self, text):
        prev = BOUNDARY
        for t, orig in tokenize(normalize(text)):
            if orig is not None:
                if t not in self.uni:
                    insort(self.sorted, t)
                    self.uni[t] = 0
                self.uni[t] += 1
                self.total += 1
                self.bi[(prev, t)] = self.bi.get((prev, t), 0) + 1
                self.ctx[prev] = self.ctx.get(prev, 0) + 1
                if orig != t and prev not in (BOUNDARY, PUNCT):
                    self.forms[t] = orig
                elif orig == t:
                    self.forms.pop(t, None)
            prev = t

    def weight(self):
        return MAX_WEIGHT * self.total / (self.total + HALF)

    def prob(self, prev, w):
        cw = self.uni.get(w)
        if not cw:
            return 0.0
        return (self.bi.get((prev, w), 0) + ALPHA * cw / self.total) / (self.ctx.get(prev, 0) + ALPHA)

    def prefixed(self, prefix, n):
        """The n most frequent user words starting with prefix."""
        lo = bisect_left(self.sorted, prefix)
        hi = bisect_left(self.sorted, prefix + "\x7f", lo)
        words = self.sorted[lo:hi]
        return words if len(words) <= n else heapq.nlargest(n, words, key=self.uni.__getitem__)

    def save(self, path=None):
        path = Path(path) if path else self.path
        if path is None:
            return
        bi = self.bi
        if len(bi) > MAX_BIGRAMS:  # too large: drop pairs seen only once
            bi = {k: c for k, c in bi.items() if c > 1}
        data = {"version": 1, "uni": self.uni, "forms": self.forms,
                "bi": [[v, w, c] for (v, w), c in bi.items()]}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.uni = {w: int(c) for w, c in data["uni"].items() if is_word(w)}
        self.forms = data.get("forms", {})
        for v, w, c in data["bi"]:
            if w in self.uni:
                self.bi[(v, w)] = int(c)
                self.ctx[v] = self.ctx.get(v, 0) + int(c)
        self.total = sum(self.uni.values())
        self.sorted = sorted(self.uni)
