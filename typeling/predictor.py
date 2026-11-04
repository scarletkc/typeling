"""
Public API.

    from typeling import Predictor
    p = Predictor()                          # bundled model; Predictor(user="user.json") enables persistent learning
    p.suggest("I'll see y")                  # -> "ou"    ghost text to display ("" when not confident)
    p.suggest("Thank you so ")               # -> "much"  after a space, predict the next word
    p.topk("I'll see you ", 3)               # -> [("again", 0.10), ...]  full words with probabilities
    p.learn("Let's meet at Starbucks.")      # learn the user's own words
    p.save()                                 # write the user model back to its file

Scoring: P(w) = (1 - lc - lu) * P_KN(w | previous two tokens) + lc * P_cache(w) + lu * P_user(w | previous token)
    cache: words among the last WINDOW tokens before the cursor (excluding the immediately preceding one),
    weighted by count; lc = cache_weight * n / (n + CACHE_HALF), where n is the number of cached words,
    because a short context says little.
Display thresholds: a next-word prediction needs P(w) >= next_threshold; completing a partial word needs
P(w | w starts with the typed letters) >= complete_threshold.
"""

import heapq
from collections import Counter
from pathlib import Path

from .model import NgramModel
from .text import BOUNDARY, normalize, split_input, tokenize
from .user import UserModel

DEFAULT_MODEL = Path(__file__).resolve().parent / "en.bin"
CONTEXT_CHARS = 2000   # only this many characters before the cursor are used
WINDOW = 300           # cache window, in tokens
CACHE_HALF = 20        # cached word count at which the cache weight reaches half of cache_weight
CANDIDATES = 8         # max candidates taken from each source (trigram, bigram, unigram, cache, user)


class _State:
    __slots__ = ("lm", "prev", "cache", "cache_total", "cache_weight", "cache_forms", "sentence_start")


class Predictor:
    def __init__(self, model=DEFAULT_MODEL, user=None, *, cache_weight=0.1, next_threshold=0.3,
                 complete_threshold=0.3, allow_blocked=False, use_context=True):
        self.lm = NgramModel(model)
        self.user = UserModel(user)
        self.cache_weight = cache_weight
        self.next_threshold = next_threshold
        self.complete_threshold = complete_threshold
        self.allow_blocked = allow_blocked
        self.use_context = use_context
        self._key, self._state = None, None

    def suggest(self, text):
        """Text before the cursor -> ghost text to display after it, or "" when not confident."""
        partial, st, ranked = self._rank(text, 1)
        if not ranked:
            return ""
        w, i, conf = ranked[0]
        if len(w) <= len(partial) or conf < (self.complete_threshold if partial else self.next_threshold):
            return ""
        return self._display(w, i, partial, st)[len(partial):]

    def topk(self, text, k=3):
        """Text before the cursor -> the k most likely full words with probabilities (no display threshold)."""
        partial, st, ranked = self._rank(text, k)
        return [(self._display(w, i, partial, st), conf) for w, i, conf in ranked]

    def learn(self, text):
        """Add text the user has typed to the user model."""
        self.user.learn(text)
        self._key = None

    def save(self, path=None):
        """Save the user model to path, or to the file given at construction."""
        self.user.save(path)

    def _context(self, text):
        """Context text -> scoring state, cached so every keystroke within a word reuses it."""
        if text == self._key:
            return self._state
        cut = text[-CONTEXT_CHARS:]
        toks = tokenize(normalize(cut))
        if len(cut) < len(text) and toks:
            toks = toks[1:]  # truncation may have cut the first word in half
        norm = [t for t, _ in toks]
        lm, st = self.lm, _State()
        if self.use_context:
            u, v = ([BOUNDARY, BOUNDARY] + norm)[-2:]
            st.lm = lm.context(lm.token_id(u), lm.token_id(v))
            window = toks[-WINDOW:]
        else:
            st.lm, window = lm.unigram_context(), []
        st.prev = norm[-1] if norm else BOUNDARY
        st.sentence_start = not norm or norm[-1] == BOUNDARY
        st.cache = Counter(t for t, o in window[:-1] if o is not None)
        st.cache_total = n = sum(st.cache.values())
        st.cache_weight = self.cache_weight * n / (n + CACHE_HALF)
        st.cache_forms = {t: o for t, o in window if o is not None and o != t}
        self._key, self._state = text, st
        return st

    def _rank(self, text, k):
        """-> (partial word, state, [(lowercase word, vocab id or None, probability)]), most probable first."""
        sp = split_input(text)
        if sp is None:
            return "", None, []
        context, partial = sp
        st = self._context(context)
        pre = partial.lower()
        lm, user = self.lm, self.user
        lo, hi = lm.prefix_range(pre)
        lc = st.cache_weight
        lu = user.weight()
        base = 1.0 - lc - lu

        cands = {}
        if hi > lo:
            for i in lm.candidates(st.lm, lo, hi, CANDIDATES):
                cands[lm.words[i]] = i
        for w in st.cache:
            if w.startswith(pre):
                cands.setdefault(w, lm.id.get(w))
        for w in user.prefixed(pre, CANDIDATES):
            cands.setdefault(w, lm.id.get(w))

        hidden = lm.misspelled if self.allow_blocked else lm.hidden
        scored, extra = [], 0.0
        for w, i in cands.items():
            if i is not None and i in hidden:
                continue
            p = 0.0
            if lc:
                p += lc * st.cache.get(w, 0) / st.cache_total
            if lu:
                p += lu * user.prob(st.prev, w)
            extra += p
            if i is not None:
                p += base * lm.prob(st.lm, i)
            if p > 0:
                scored.append((p, w, i))
        if not scored:
            return partial, st, []
        top = heapq.nlargest(k, scored)
        z = 1.0
        if pre:  # completion: condition on the word starting with the typed letters
            z = (base * lm.range_mass(st.lm, lo, hi) if hi > lo else 0.0) + extra
            z = max(z, sum(p for p, _, _ in scored))
        return partial, st, [(w, i, p / z) for p, w, i in top]

    def _display(self, w, i, partial, st):
        """Lowercase word -> display form: keep the letters already typed; continue in the word's usual form
        if the typed letters match its casing (iPh -> iPhone), in lowercase otherwise (ng -> ngos, not ngOs),
        and in capitals if the typed letters are all caps. Capitalize at sentence start."""
        form = self.user.forms.get(w) or (self.lm.form(i) if i is not None else st.cache_forms.get(w, w))
        if partial:
            tail = form[len(partial):]
            if len(partial) > 1 and partial.isupper():
                return partial + tail.upper()
            return partial + (tail if form.startswith(partial) else tail.lower())
        return form[:1].upper() + form[1:] if st.sentence_start else form
