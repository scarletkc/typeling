# /// script
# requires-python = ">=3.10"
# dependencies = ["wordfreq"]
# ///
"""
从 wordfreq 生成词表 words.txt（每行：单词<TAB>Zipf 词频）。
只需运行一次，wordfreq 由 uv 装在临时环境里，不污染全局：

    uv run build_vocab.py            # 默认取 50000 个词
    uv run build_vocab.py 100000     # 指定词数
"""

import re
import sys
from importlib.metadata import version

from wordfreq import top_n_list, zipf_frequency

MAX_LEN = 20
# 缩写拆出来的碎片，不算完整单词（单字母只保留 a 和 i）
FRAGMENTS = {"s", "t", "d", "m", "re", "ll", "ve", "nt"}
VALID = re.compile(r"^[a-z]+$")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    n =int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    words = []
    for w in top_n_list("en", n * 3):
        if not VALID.match(w) or len(w) > MAX_LEN:
            continue
        if w in FRAGMENTS or (len(w) == 1 and w not in ("a", "i")):
            continue
        words.append((w, zipf_frequency(w, "en")))
        if len(words) == n:
            break

    with open("words.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# source: wordfreq {version('wordfreq')} (data: CC BY-SA 4.0)\n")
        f.write("# format: word<TAB>zipf  (zipf = log10(每十亿词出现次数))\n")
        for w, z in words:
            f.write(f"{w}\t{z:.2f}\n")
    print(f"写出 {len(words)} 个词到 words.txt")


if __name__ == "__main__":
    main()
