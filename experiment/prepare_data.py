# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow"]
# ///
"""
下载 WikiText-2（raw）并整理成纯文本，只需运行一次：

    uv run experiment/prepare_data.py

产物（都在 data/ 下，已被 .gitignore 忽略）：
    data/raw/*.parquet              原始下载
    data/wikitext2/train.txt        训练集：每行一个段落
    data/wikitext2/valid.txt        开发集
    data/wikitext2/test.txt         测试集（封存：evaluate.py 不加 --final 不会读它）
    data/wikitext2/vocab.txt        共享词表：训练集里出现 ≥2 次的纯字母词，按字母序，不含次数

数据来源：Hugging Face Salesforce/wikitext，wikitext-2-raw-v1，CC BY-SA 3.0
"""

import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "wikitext2"
BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
SPLITS = {"train": "train", "valid": "validation", "test": "test"}
WORD = re.compile(r"^[a-z]+$")
MIN_COUNT = 2


def download(name):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.parquet"
    if not path.exists():
        url = f"{BASE}{name}-00000-of-00001.parquet"
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, path)
    return path


def paragraphs(path):
    """去掉空行和 = 标题 = 行；把 @-@ @,@ @.@ 还原成普通符号（仍保留空格分词）。"""
    out = []
    for line in pq.read_table(path).column("text").to_pylist():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        line = line.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
        out.append(line)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for split, hf_name in SPLITS.items():
        paras = paragraphs(download(hf_name))
        (OUT / f"{split}.txt").write_text("\n".join(paras) + "\n", encoding="utf-8")
        tokens = [t for p in paras for t in p.lower().split()]
        words = [t for t in tokens if WORD.match(t)]
        print(f"{split:<6} {len(paras):>6} 段  {len(tokens):>9,} 个 token  {len(words):>9,} 个纯字母词")
        if split == "train":
            counts.update(words)
    vocab = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
    (OUT / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    print(f"共享词表 {len(vocab):,} 个词（训练集出现 ≥{MIN_COUNT} 次）")


if __name__ == "__main__":
    main()
