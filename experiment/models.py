"""
参赛模型的统一接口和注册表。

每个模型实现四个方法：
    begin_word(context)  每个目标词开始时调用一次；context 是同一段落里前面所有 token（小写）
    predict(prefix, k)   每"打"一个字母调用一次；prefix 是目标词已输入的字母（可为空串），
                         返回最多 k 个完整单词，按可能性从高到低
    knows(word)          词是否在模型词表里（用来统计 OOV）
    size_bytes()         加载模型需要的文件总大小（可以是压缩后的）

新模型写好后在 REGISTRY 里登记一行即可：python experiment/evaluate.py <名字>
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Model:
    name = "base"
    kind = "baseline"   # baseline / H0 / H1 / H2 / T
    desc = ""

    def begin_word(self, context):
        pass

    def predict(self, prefix, k):
        raise NotImplementedError

    def knows(self, word):
        return True

    def size_bytes(self):
        return 0


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


REGISTRY = {
    "B0": B0Unigram,
}
