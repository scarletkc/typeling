"""
Typing decoder: word suggestions from a causal subword language model, with the same interface as
typeling.Predictor.

    from decode import LMPredictor
    p = LMPredictor("SupraLabs/Supra-50M-Base")
    p.topk("Let me know ", 3)    # -> [("if", 0.21), ...]    next word: P(word | context, the space)
    p.topk("Let me kn", 3)       # -> [("know", 0.98), ...]  completion: P(word | context, the typed letters)
    p.suggest("Let me kn")       # -> "ow"                   ghost text, "" when not confident

Decoding: the text before the cursor is split into the context, the whitespace after it (the lead) and the partially
typed word. Byte-level BPE attaches the space to the following word (" know"), so a context ending in a space would
force the model to continue with a rare token. The context is therefore cut before the trailing whitespace, and the
lead plus the typed letters (" kn") become a constraint on the generated text:
    prefix phase  only tokens consistent with the constraint (" k", " kn", " know", ...)
    word phase    the word grows with letter tokens ("know" + "ing", "don" + "'t") until the next token is anything
                  else (a space, punctuation, end of text), whose total probability ends the word
A word's probability is the sum, over its token paths, of path probability x end probability, divided by the
probability of the constraint itself. Paths are explored with a small beam; casings of a word are merged. Words
follow typeling.text.WORD (ASCII letters with inner apostrophes).
Supports GPT-2-style byte-level BPE tokenizers (Llama-architecture models such as Supra, GPT-NeoX/Pythia, SmolLM2).
"""

import bisect
import math
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from typeling.text import normalize, split_input  # noqa: E402

CONTEXT_CHARS = 2000   # only this many characters before the cursor are used, as in Predictor
WORD_STATE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]*)*")  # a word being built; may end in an apostrophe for now
LETTERS = re.compile(r"[A-Za-z]+")
APOSTROPHE = re.compile(r"'[A-Za-z]*")                 # "'" or a contraction token such as "'t"
NEVER = "\x00"                                          # string of special and undecodable tokens


def _byte_decoder():
    """GPT-2 byte-level BPE alphabet: printable character -> byte."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    rest = [b for b in range(256) if b not in printable]
    return {chr(b): b for b in printable} | {chr(256 + i): b for i, b in enumerate(rest)}


class LMPredictor:
    def __init__(self, model, *, device=None, dtype=torch.float32, start=None, beam=8, max_word_tokens=6,
                 next_threshold=0.3, complete_threshold=0.3):
        """model: Hugging Face repo id or local directory. start: text that marks the beginning of a text
        (default: the tokenizer's EOS token, which separates the texts in Typeling's training data)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.params = sum(t.numel() for t in self.model.parameters())
        self.beam, self.max_word_tokens = beam, max_word_tokens
        self.next_threshold, self.complete_threshold = next_threshold, complete_threshold
        self.start = start if start is not None else (self.tok.eos_token or self.tok.bos_token or "\n")
        self.start_ids = self.encode(self.start)
        self.max_context = min(getattr(self.model.config, "max_position_embeddings", 1024), 1024) - 64
        self._index_vocab()
        self.prefix_space = self.strs[self.encode("a")[0]].startswith(" ")  # tokenizer adds a space to every text
        self._base = self._ids = self._cache = self._logp0 = None

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _index_vocab(self):
        """Token strings, a sorted index for prefix lookups, and masks of word-continuing tokens."""
        n = self.model.get_output_embeddings().weight.shape[0]
        dec = _byte_decoder()
        special = set(self.tok.all_special_ids) | {i for i in self.tok.added_tokens_decoder if i < n}
        strs = []
        for i in range(n):
            piece = self.tok.convert_ids_to_tokens(i) if i < len(self.tok) else None
            if i in special or not piece or any(c not in dec for c in piece):
                strs.append(NEVER)
            else:
                strs.append(bytes(dec[c] for c in piece).decode("utf-8", errors="replace"))
        self.strs = strs
        self._order = sorted(range(n), key=strs.__getitem__)
        self._sorted = [strs[i] for i in self._order]
        self._sorted_ids = torch.tensor(self._order, device=self.device)
        self._by_str = {}
        for i, s in enumerate(strs):
            self._by_str.setdefault(s, i)
        letters = torch.tensor([bool(LETTERS.fullmatch(s)) for s in strs], device=self.device)
        apostrophe = torch.tensor([bool(APOSTROPHE.fullmatch(s)) for s in strs], device=self.device)
        self._cont = letters | apostrophe   # continues a word
        self._letters = letters             # continues a word that ends in an apostrophe

    # --- Public API (same as typeling.Predictor) ---

    def topk(self, text, k=3):
        """Text before the cursor -> the k most likely full words with probabilities (no display threshold)."""
        return [(w, p) for w, p in self._rank(text, k)[1]]

    def suggest(self, text):
        """Text before the cursor -> ghost text to display after it, or "" when not confident."""
        partial, ranked = self._rank(text, 1)
        if not ranked:
            return ""
        w, conf = ranked[0]
        if len(w) <= len(partial) or conf < (self.complete_threshold if partial else self.next_threshold):
            return ""
        return w[len(partial):]

    def learn(self, text):
        """No user model yet; present for interface compatibility."""

    # --- Context ---

    def _context(self, base):
        """Run the model over the context, reusing the key/value cache of the longest common token prefix."""
        if base == self._base:
            return
        cut = base[-CONTEXT_CHARS:]
        if len(cut) < len(base):
            ids = self.encode(cut[cut.find(" ") + 1:] if " " in cut else cut)  # drop the first, cut word
        else:
            ids = self.start_ids + self.encode(base)
        ids = ids[-self.max_context:]
        n = 0
        if self._ids is not None:
            for a, b in zip(ids, self._ids):
                if a != b:
                    break
                n += 1
        n = min(n, len(ids) - 1)  # at least one token is fed, for its next-token distribution
        if n == 0:
            self._cache = DynamicCache(config=self.model.config)
        else:
            self._truncate(n)
        out = self.model(input_ids=torch.tensor([ids[n:]], device=self.device), past_key_values=self._cache,
                         use_cache=True)
        self._logp0 = torch.log_softmax(out.logits[0, -1].float(), -1)
        self._base, self._ids = base, ids

    def _truncate(self, length):
        """Keep the first length positions of the key/value cache (crop() takes the number to remove)."""
        extra = self._cache.get_seq_length() - length
        if extra > 0:
            self._cache.crop(-extra)

    def _range(self, prefix):
        """Index range in self._sorted of the tokens that start with prefix."""
        return bisect.bisect_left(self._sorted, prefix), bisect.bisect_left(self._sorted, prefix + "\U0010ffff")

    # --- Search ---

    @torch.inference_mode()
    def _rank(self, text, k):
        """-> (partial word, [(word, probability)]), most probable first."""
        sp = split_input(normalize(text))
        if sp is None:
            return "", []
        context, partial = sp
        base = context.rstrip()
        lead = context[len(base):]
        if not base and not lead and self.prefix_space:
            lead = " "  # such a model starts every text with a space, so the first word is " word"
        self._context(base)
        target, start = lead + partial, len(lead)
        beam, strs = self.beam, self.strs
        words = {}   # lowercase word -> [probability, most probable form, its probability]
        z = 0.0      # probability of the constraint (lead + typed letters)
        hyps = [("", 0.0)]   # (generated text, log probability); row i of logp
        logp = self._logp0[None]
        try:
            for _ in range(len(target) + self.max_word_tokens):
                new = []   # (log probability, parent row, token id, text)
                rows = [i for i, (g, _) in enumerate(hyps) if len(g) >= len(target)]
                if rows:   # word phase, vectorized over rows
                    lp = logp[rows]
                    p = lp.exp()
                    apos = torch.tensor([hyps[i][0].endswith("'") for i in rows], device=self.device)[:, None]
                    mask = torch.where(apos, self._letters, self._cont)
                    ends = (1.0 - (p * mask).sum(-1)).clamp_min(0.0).tolist()
                    top = torch.topk(lp.masked_fill(~mask, -math.inf), beam)
                    for r, row in enumerate(rows):
                        g, glp = hyps[row]
                        word = g[start:].rstrip("'")
                        e = words.setdefault(word.lower(), [0.0, word, 0.0])
                        q = math.exp(glp) * ends[r]
                        e[0] += q
                        if q > e[2]:
                            e[1], e[2] = word, q
                        for v, t in zip(top.values[r].tolist(), top.indices[r].tolist()):
                            if v > -math.inf:
                                new.append((glp + v, row, t, g + strs[t]))
                for row, (g, glp) in enumerate(hyps):
                    if len(g) >= len(target):
                        continue
                    rest = target[len(g):]   # prefix phase
                    lrow = logp[row]
                    lo, hi = self._range(rest)
                    if hi > lo:
                        vals = lrow[self._sorted_ids[lo:hi]]
                        z += math.exp(glp) * vals.exp().sum().item()
                        top = torch.topk(vals, min(3 * beam, hi - lo))
                        for v, j in zip(top.values.tolist(), top.indices.tolist()):
                            t = self._order[lo + j]
                            s = g + strs[t]
                            if WORD_STATE.fullmatch(s, start):
                                new.append((glp + v, row, t, s))
                    for i in range(1, len(rest)):
                        t = self._by_str.get(rest[:i])
                        if t is not None:
                            new.append((glp + lrow[t].item(), row, t, g + rest[:i]))
                if not new:
                    break
                new.sort(reverse=True)
                new = new[:beam]
                best = sorted((e[0] for e in words.values()), reverse=True)
                if len(best) >= k and math.exp(new[0][0]) < best[k - 1]:
                    break
                self._cache.reorder_cache(torch.tensor([h[1] for h in new], device=self.device))
                out = self.model(input_ids=torch.tensor([[h[2]] for h in new], device=self.device),
                                 past_key_values=self._cache, use_cache=True)
                logp = torch.log_softmax(out.logits[:, -1].float(), -1)
                hyps = [(h[3], h[0]) for h in new]
        finally:
            self._truncate(len(self._ids))
            self._cache.reorder_cache(torch.tensor([0], device=self.device))
        if z <= 0:
            return partial, []
        ranked = sorted(((e[0] / z, e[1]) for e in words.values() if e[0] > 0), reverse=True)[:k]
        return partial, [(w, p) for p, w in ranked]
