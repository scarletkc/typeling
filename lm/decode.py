"""
Experimental typed mode on top of the packaged typing decoder (typeling.lm), for models fine-tuned on typos
(lm/typos.py, an experiment for v1.1; the tokenizer has <|typed|> and <|word|>). A partially typed word is decoded
from "context <|typed|> typed letters <|word|>" instead, as a word token sequence ended by </s>; the probabilities are
already conditioned on the typed letters, and the word need not start with them (recie -> receive). Models without
the two tokens decode exactly as with typeling.lm.LMPredictor.

    from decode import TypedPredictor
    p = TypedPredictor("data/lm/runs/typo-40m")
"""

import math
import re
import sys
from pathlib import Path

import torch

from typos import TYPO_TOKENS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from typeling.lm import EXACT, LMPredictor  # noqa: E402

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")   # a finished word (typeling.text.WORD)
WORD_START = re.compile(r" [A-Za-z]+")           # a token that starts a word mid-text


class TypedPredictor(LMPredictor):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        vocab = self.tok.get_vocab()
        self.typed_ids = [vocab[t] for t in TYPO_TOKENS] if all(t in vocab for t in TYPO_TOKENS) else None
        if self.typed_ids:
            self._word_start = torch.tensor([bool(WORD_START.fullmatch(s)) for s in self.strs], device=self.device)

    def _search(self, context, partial, k):
        if partial and self.typed_ids:
            return self._rank_typed(context.rstrip(), partial, k)
        return super()._search(context, partial, k)

    @torch.inference_mode()
    def _rank_typed(self, base, partial, k):
        """Typed mode: decode the intended word from "context <|typed|> partial <|word|>", ended by </s>."""
        self._context(base)
        typed_id, word_id = self.typed_ids
        letters = self.encode(partial if self.prefix_space else " " + partial)
        eos, beam, strs, exact = self.tok.eos_token_id, self.beam, self.strs, min(k, EXACT)
        words = {}   # lowercase word -> [probability, most probable form, its probability]
        try:
            out = self.model(input_ids=torch.tensor([[typed_id, *letters, word_id]], device=self.device),
                             past_key_values=self._cache, use_cache=True)
            logp = torch.log_softmax(out.logits[:, -1].float(), -1)
            hyps = [("", 0.0)]   # (generated text, log probability); row i of logp
            for _ in range(self.max_word_tokens + 1):
                started = torch.tensor([bool(g) for g, _ in hyps], device=self.device)[:, None]
                apos = torch.tensor([g.endswith("'") for g, _ in hyps], device=self.device)[:, None]
                mask = torch.where(started, torch.where(apos, self._letters, self._cont), self._word_start)
                ends = logp[:, eos].tolist()
                top = torch.topk(logp.masked_fill(~mask, -math.inf), beam)
                new = []   # (log probability, parent row, token id, text)
                for r, (g, glp) in enumerate(hyps):
                    word = g[1:]
                    if WORD.fullmatch(word):
                        q = math.exp(glp + ends[r])
                        e = words.setdefault(word.lower(), [0.0, word, 0.0])
                        e[0] += q
                        if q > e[2]:
                            e[1], e[2] = word, q
                    for v, t in zip(top.values[r].tolist(), top.indices[r].tolist()):
                        if v > -math.inf:
                            new.append((glp + v, r, t, g + strs[t]))
                if not new:
                    break
                new.sort(reverse=True)
                new = new[:beam]
                best = sorted((e[0] for e in words.values()), reverse=True)
                if len(best) >= exact and math.exp(new[0][0]) < best[exact - 1]:
                    break
                self._cache.reorder_cache(torch.tensor([h[1] for h in new], device=self.device))
                out = self.model(input_ids=torch.tensor([[h[2]] for h in new], device=self.device),
                                 past_key_values=self._cache, use_cache=True)
                logp = torch.log_softmax(out.logits[:, -1].float(), -1)
                hyps = [(h[3], h[0]) for h in new]
        finally:
            self._truncate(len(self._ids))
            self._cache.reorder_cache(torch.tensor([0], device=self.device))
        ranked = sorted(((e[0], e[1]) for e in words.values() if e[0] > 0), reverse=True)[:k]
        return [(w, p) for p, w in ranked]
