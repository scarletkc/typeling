"""
对照组：现成的预训练小语言模型 Pythia-14M（EleutherAI，1400 万参数，28 MB，Apache-2.0，训练语料 the Pile）。

    uv run --no-project --with torch --with transformers python experiment/evaluate.py P14M

第一次运行会把 model.safetensors 和分词器文件下载到 data/models/pythia-14m/（已被 .gitignore 忽略）。

子词模型 → 整词补全的打分（版本 approx2）：
    词表里每个词 w 取 4 种写法：w / W（首字母大写）× 前面有没有空格（按原文实际排版选其中 2 种）
    score(w) = log Σ_写法 P(第 1 个子词 | 段落前文) · P(后面的子词 | 第 1 个子词) · P(词到此结束 | 这些子词)
    第 1 个子词的概率按真实前文精确计算；后面两项用"不看前文"的条件概率近似（预先算好）。
    "词到此结束" = 下一个子词以空格或标点开头；不乘这一项的话，" M" 这种子词会把所有 M 开头的长词的
    概率都算到单字母词 m 头上。
    这样对多子词的罕见词偏保守（拿不到前文里重复出现同一个词的好处），所以得分是 Pythia 的偏低估计。
段落按原始大小写、还原成正常排版（去掉 WikiText 的空格分词）后送进模型；只看本段前文，和其他模型一致。
"""

import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import ROOT, ScoreModel, load_shared_vocab

REPO = "EleutherAI/pythia-14m"
MODEL_DIR = ROOT / "data" / "models" / "pythia-14m"
FILES = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
WORD_RE = re.compile(r"^[a-z]+$")
NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "'s", "n't", "'re", "'ve", "'ll", "'m", "'d", "'"}
NO_SPACE_AFTER = {"(", "[", "{", "$"}
MAX_CTX = 2048


def ensure_downloaded():
    if all((MODEL_DIR / f).exists() for f in FILES):
        return
    from huggingface_hub import snapshot_download

    print(f"下载 {REPO} 到 {MODEL_DIR} …")
    snapshot_download(REPO, local_dir=MODEL_DIR, allow_patterns=FILES)


def detokenize(toks):
    """WikiText 空格分词 → 正常排版。返回文本和每个 token 的起始字符位置。"""
    parts, starts, length, quote_open = [], [], 0, False
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else None
        if i == 0 or t in NO_SPACE_BEFORE or t == "-" or prev == "-" or prev in NO_SPACE_AFTER:
            sep = ""
        elif t == '"':
            sep = "" if quote_open else " "
        elif prev == '"' and quote_open:
            sep = ""
        else:
            sep = " "
        if t == '"':
            quote_open = not quote_open
        parts.append(sep)
        length += len(sep)
        starts.append(length)
        parts.append(t)
        length += len(t)
    return "".join(parts), starts


class PythiaRef(ScoreModel):
    name = "P14M"
    kind = "ref"
    version = "approx2"
    desc = "现成预训练模型 Pythia-14M（1400 万参数）：首个子词按前文精确算，后续子词用无前文近似"

    def __init__(self):
        super().__init__(load_shared_vocab())
        ensure_downloaded()
        torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.lm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.F, self.C = self._vocab_tables()
        self.rows, self.lp = {}, None
        self.n_targets = self.n_unmapped = 0

    def size_bytes(self):
        return (MODEL_DIR / "model.safetensors").stat().st_size

    def notes(self):
        return (f"有 {self.n_unmapped}/{self.n_targets} 个目标词在子词切分里找不到干净的词首边界，按"
                f"'没有候选'计")

    # ───────── 词表：每种写法的第 1 个子词和"无前文"后续概率（算一次，缓存） ─────────

    @torch.no_grad()
    def _vocab_tables(self):
        """F[写法, 词] = 第 1 个子词；C[写法, 词] = log P(后面的子词) + log P(词到此结束)，都在无前文下算。"""
        cache = MODEL_DIR / "vocab_tables_v2.npz"
        if cache.exists():
            d = np.load(cache)
            if d["F"].shape[1] == len(self.words):
                return d["F"], d["C"]
        V = len(self.words)
        n_vocab = self.lm.get_output_embeddings().weight.shape[0]
        pieces = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        boundary = torch.zeros(n_vocab, dtype=torch.bool)  # 以空格 / 标点开头的子词 = 上一个词结束了
        for t, p in enumerate(pieces):
            s = self.tok.convert_tokens_to_string([p]) if p is not None else ""
            boundary[t] = not s or not s[0].isalpha()
        forms = [lambda w: " " + w, lambda w: " " + w.capitalize(), lambda w: w, lambda w: w.capitalize()]
        F = np.zeros((4, V), dtype=np.int64)
        C = np.zeros((4, V), dtype=np.float32)
        by_len = {}
        for k, form in enumerate(forms):
            enc = self.tok([form(w) for w in self.words], add_special_tokens=False)["input_ids"]
            for i, ids in enumerate(enc):
                F[k, i] = ids[0]
                by_len.setdefault(len(ids), []).append((k, i, ids))
        for n, items in by_len.items():
            for s in range(0, len(items), 2048):
                chunk = items[s:s + 2048]
                x = torch.tensor([[self.bos] + ids for _, _, ids in chunk])
                logp = torch.log_softmax(self.lm(x).logits.float(), dim=-1)
                cont = logp[:, 1:-1].gather(-1, x[:, 2:].unsqueeze(-1)).squeeze(-1).sum(-1)  # 第 2 个子词起
                end = torch.logsumexp(logp[:, -1][:, boundary], dim=-1)
                for (k, i, _), v in zip(chunk, (cont + end).numpy()):
                    C[k, i] = v
        np.savez(cache, F=F, C=C)
        return F, C

    # ───────── 每段一次：整段前向，取出每个目标词位置的"下一个子词"分布 ─────────

    @torch.no_grad()
    def begin_paragraph(self, orig):
        text, starts = detokenize(orig)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        start_to_bpe = {}
        for b, (s, e) in enumerate(offs):
            start_to_bpe.setdefault(s, b)
            if s < e and text[s] == " ":  # 有的分词器把前导空格算进偏移
                start_to_bpe.setdefault(s + 1, b)
        need = {}
        for j, t in enumerate(orig):
            if WORD_RE.match(t.lower()):
                self.n_targets += 1
                b = start_to_bpe.get(starts[j])
                if b is None:
                    self.n_unmapped += 1
                    continue
                spaced = starts[j] > 0 and text[starts[j] - 1] == " "
                need[j] = (b, spaced)  # 预测第 b 个子词 = 看 [BOS] + ids[:b] 最后一个位置
        x = [self.bos] + ids
        hidden = self._hidden(x)
        pos = sorted({b for b, _ in need.values()})
        row_of = {b: r for r, b in enumerate(pos)}
        if pos:
            h = hidden[torch.tensor(pos)]
            self.lp = torch.log_softmax(self.lm.get_output_embeddings()(h).float(), dim=-1).numpy()
        self.rows = {j: (row_of[b], spaced) for j, (b, spaced) in need.items()}

    def _hidden(self, x):
        """最后一层隐状态；超长段落分窗口算（每个位置至少带上一半窗口的前文）。"""
        if len(x) <= MAX_CTX:
            return self.lm.base_model(torch.tensor([x])).last_hidden_state[0]
        out = torch.empty(len(x), self.lm.config.hidden_size)
        half = MAX_CTX // 2
        start = 0
        while True:
            h = self.lm.base_model(torch.tensor([x[start:start + MAX_CTX]])).last_hidden_state[0]
            keep_from = 0 if start == 0 else half
            out[start + keep_from:start + len(h)] = h[keep_from:]
            if start + MAX_CTX >= len(x):
                return out
            start += half

    def context_scores(self, context):
        hit = self.rows.get(len(context))
        if hit is None:
            return None
        row, spaced = hit
        lp = self.lp[row]
        k = 0 if spaced else 2
        return np.logaddexp(lp[self.F[k]] + self.C[k], lp[self.F[k + 1]] + self.C[k + 1])

    def predict(self, prefix, k):
        if self.scores is None:
            return []
        return super().predict(prefix, k)
