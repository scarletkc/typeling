"""
参赛模型的统一接口和注册表。

每个模型实现四个方法：
    begin_word(context)  每个目标词开始时调用一次；context 是同一段落里前面所有 token（小写）
    predict(prefix, k)   每"打"一个字母调用一次；prefix 是目标词已输入的字母（可为空串），
                         返回最多 k 个完整单词，按可能性从高到低
    knows(word)          词是否在模型词表里（用来统计 OOV）
    size_bytes()         模型自己的文件总大小（共享词表 vocab.txt 不计）

新模型写好后在 REGISTRY 里登记一行即可：python experiment/evaluate.py <名字>
"""

import bisect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "wikitext2" / "vocab.txt"
sys.path.insert(0, str(ROOT))


class Model:
    name = "base"
    kind = "baseline"   # baseline / H0 / H1 / H2 / T
    version = None
    desc = ""

    def begin_word(self, context):
        pass

    def predict(self, prefix, k):
        raise NotImplementedError

    def knows(self, word):
        return True

    def size_bytes(self):
        return 0


class ScoreModel(Model):
    """先给整个词表打一次上下文分数（每个词开始时一次），再按前缀取最高的 k 个（每键一次）。

    以某个前缀开头的词在字母序里连成一段，所以"前缀筛选"只需在排好序的词表里二分出区间——
    这和 B0 那层手写的 W·x 精确匹配完全等价，只是更快。子类只需实现 context_scores。
    """

    def __init__(self, words):
        self.words = words  # 已按字母序排好
        self.index = {w: i for i, w in enumerate(words)}
        self.scores = None

    def context_scores(self, context):
        raise NotImplementedError

    def begin_word(self, context):
        self.scores = self.context_scores(context)

    def predict(self, prefix, k):
        lo = bisect.bisect_left(self.words, prefix)
        hi = bisect.bisect_left(self.words, prefix + "{", lo)  # "{" 紧排在 "z" 之后
        if lo >= hi:
            return []
        seg = self.scores[lo:hi]  # 视图：下面临时改值，取完再还原
        if hi - lo <= k:
            return [self.words[lo + i] for i in np.argsort(-seg, kind="stable")]
        picked, saved = [], []
        for _ in range(k):
            i = int(seg.argmax())
            picked.append(i)
            saved.append(seg[i])
            seg[i] = -np.inf
        for i, v in zip(picked, saved):
            seg[i] = v
        return [self.words[lo + i] for i in picked]

    def knows(self, word):
        return word in self.index


def load_shared_vocab():
    if not VOCAB_PATH.exists():
        sys.exit(f"找不到 {VOCAB_PATH}，先运行：uv run experiment/prepare_data.py")
    return VOCAB_PATH.read_text(encoding="utf-8").split()


class B0Unigram(Model):
    """现有补全内核：前缀匹配 + wordfreq 词频，不看上下文。"""

    name = "B0"
    kind = "baseline"
    desc = "现有内核：W·x + b，只看前缀和 wordfreq 词频，不看上下文"

    def __init__(self):
        from complete_kernel import DEFAULT_WEIGHTS, Completer

        self.c = Completer()
        self.path = DEFAULT_WEIGHTS
        order = np.argsort(-self.c.b, kind="stable")
        self.top_empty = [self.c.words[i] for i in order[:20]]
        self.vocab = set(self.c.words)

    def predict(self, prefix, k):
        if not prefix:  # 刚打完空格：没有前缀，只能按词频猜
            return self.top_empty[:k]
        return self.c.candidates(prefix, k)

    def knows(self, word):
        return word in self.vocab

    def size_bytes(self):
        return self.path.stat().st_size


def _h2():
    from h2_handwritten import H2Model

    return H2Model()


def _p14m():  # 需要 torch：uv run --no-project --with torch --with transformers python experiment/evaluate.py P14M
    from lm_reference import PythiaRef

    return PythiaRef()


REGISTRY = {
    "B0": B0Unigram,
    "H2": _h2,
    "P14M": _p14m,
}
