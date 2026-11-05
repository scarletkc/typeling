"""
Typing decoder for a small causal language model (Typeling-25M or another byte-level BPE model), with the same
interface as typeling.Predictor. Needs PyTorch and Transformers: pip install "typeling[lm]".

    from typeling.lm import LMPredictor
    p = LMPredictor()                   # Typeling-25M from Hugging Face; or a repo id / local directory
    p.topk("Let me know ", 3)           # -> [("if", 0.73), ("how", 0.12), ...]  next word
    p.topk("Let me kn", 3)              # -> [("know", 1.0), ...]                completion
    p.topk("See you tomm", 3)           # -> [("tomorrow", 0.9), ...]            completion with typo repair
    p.suggest("Let me kn")              # -> "ow"   ghost text, "" when not confident

Candidates: topk()'s search runs until no hypothesis left in the beam can beat the EXACT-th word found so far;
candidates after it are the other words the search reached, most probable first. None of them is exact: paths pruned
from the beam are not revisited, and hidden words, the user model and n-gram fusion are applied after the search.

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

Typo repair (typo_weight > 0, for partial words of typo_min+ letters): the typed letters may contain one typing
error (typeling.keyboard.corrections(): adjacent key, extra key, left-out letter, swapped letters, missing capital).
The corrections that are prefixes of words in the bundled n-gram model's vocabulary are ranked by the unigram mass of
the words they start, and the best typo_alts become extra constraints, searched in the same beam as the typed
letters. Each starts with log(typo_weight) added to its path probabilities, so a word's probability becomes
    sum over constraints c it satisfies of weight(c) x P(word | context)  /  sum over c of weight(c) x P(c | context)
(noisy channel, weight 1 for the letters as typed, or unknown_weight if they start no vocabulary word: the model may
have learned common misspellings such as "recieve" and would otherwise prefer them). topk() may then return words
that do not start with the typed letters (tomm -> tomorrow); suggest() shows only extensions.

N-gram fusion (ngram_weight > 0): the final probability is (1 - ngram_weight) x the language model's + ngram_weight x
the bundled n-gram model's (typeling.Predictor, its top NGRAM_CANDIDATES words, 0 for the others), so it adds at most
ngram_weight to any word. It pushes down non-words that the language model's subword pieces can form (don'tknow,
muchfor).

User model (user=path; learn(), save()): the words and word pairs the user typed, in typeling.Predictor's format
(typeling.user.UserModel). It is mixed into the joint probabilities before they are normalized over the words that
fit the typed letters:
    P(word | context, typed letters) is proportional to (1 - u) x P_LM(word, typed letters | context)
                                                        + u x P_user(word | previous word)
with u = UserModel.weight(), which grows with the amount learned (up to 0.3). A word that the language model barely
knows (a name) takes the lead after the user has typed it a few times, because the language model gives its typed
letters almost no probability; common words stay with the language model. Learned words also count as known words
for unknown_weight.

As in typeling.Predictor, the n-gram model's known misspellings (recieve) are never suggested, nor its offensive words
unless allow_blocked is set; the language model's probabilities are renormalized without them.
"""

import bisect
import math
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
except ImportError as e:
    raise ImportError('typeling.lm needs PyTorch and Transformers: pip install "typeling[lm]"') from e

from .keyboard import corrections
from .predictor import Predictor
from .text import normalize, split_input
from .user import UserModel

DEFAULT_LM = "ScarletKc/Typeling-25M"
EXACT = 3              # topk() searches until this many candidates are settled within the beam
NGRAM_CANDIDATES = 8   # n-gram words fused per query
CONTEXT_CHARS = 1000   # default: at most this many characters before the cursor are used (2,000 cost 0.1 KSR
                       # points less on long texts but up to 40% more time per query on a CPU)
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
    def __init__(self, model=DEFAULT_LM, user=None, *, device=None, dtype=torch.float32, start=None, beam=8,
                 max_word_tokens=6, next_threshold=0.3, complete_threshold=0.3, typo_weight=0.003, typo_min=2,
                 typo_alts=4, unknown_weight=0.1, ngram_weight=0.1, allow_blocked=False, context_chars=CONTEXT_CHARS):
        """model: Hugging Face repo id or local directory. user: user model file for persistent learning (as in
        typeling.Predictor). device: "cpu" or "cuda" (default: cuda if available).
        start: text that marks the beginning of a text (default: the tokenizer's EOS token, which separates the
        texts in Typeling's training data). typo_weight: probability of a typing error relative to none, 0 to turn
        typo repair off. unknown_weight: weight of the letters as typed when they start no vocabulary word and
        corrections exist. ngram_weight: share of the bundled n-gram model in the probabilities, 0 to turn it off.
        allow_blocked: also suggest the n-gram model's offensive words. context_chars: at most this many characters
        before the cursor are used (fewer make long texts faster)."""
        self.context_chars = context_chars
        self.typo_weight, self.typo_min, self.typo_alts = typo_weight, typo_min, typo_alts
        self.unknown_weight = unknown_weight
        self.ngram_weight = ngram_weight
        self.ngram = Predictor(allow_blocked=allow_blocked)
        self.user = UserModel(user)
        self._load_lexicon(allow_blocked)
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
        self._last = None   # (text, k, partial, ranked) of the last search

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
        """Text before the cursor -> up to k full words with probabilities, most probable first (no display
        threshold). With typo repair, words need not start with the typed letters."""
        return self._rank(text, k)[1][:k]

    def suggest(self, text):
        """Text before the cursor -> ghost text to display after it, or "" when not confident."""
        partial, ranked = self._rank(text, 1)
        if not ranked:
            return ""
        w, conf = ranked[0]
        if (len(w) <= len(partial) or not w.lower().startswith(partial.lower())
                or conf < (self.complete_threshold if partial else self.next_threshold)):
            return ""
        return w[len(partial):]

    def learn(self, text):
        """Add text the user has typed to the user model."""
        self.user.learn(text)
        self._last = None

    def save(self, path=None):
        """Save the user model to path, or to the file given at construction."""
        self.user.save(path)

    def _rank(self, text, k):
        """-> (partial word, [(word, probability)]), most probable first; repeated calls for the same text reuse
        the last search."""
        if self._last and self._last[0] == text and self._last[1] >= k:
            return self._last[2], self._last[3]
        sp = split_input(normalize(text))
        partial, ranked = (sp[1], self._search(*sp, k)) if sp else ("", [])
        self._last = (text, k, partial, ranked)
        return partial, ranked

    # --- Context ---

    def _context(self, base):
        """Run the model over the context, reusing the key/value cache of the longest common token prefix."""
        if base == self._base:
            return
        # the start of a cut context moves in quarter steps, so that the key/value cache stays valid in between
        # (cutting at exactly context_chars would shift it on every word)
        step = max(1, self.context_chars // 4)
        start = -(-max(0, len(base) - self.context_chars) // step) * step
        if start:
            cut = base[start:]
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

    # --- Typo repair ---

    def _load_lexicon(self, allow_blocked):
        """Sorted lowercase words of the bundled n-gram model, cumulative unigram probabilities (0 for its blocked
        words and known misspellings) and the words never suggested (known misspellings such as recieve, definately;
        offensive words unless allowed)."""
        lm = self.ngram.lm
        self._lex = lm.words
        self._hidden = {lm.words[i] for i in (lm.misspelled if allow_blocked else lm.hidden)}
        self._lex_cum = [0.0]
        for i, p in enumerate(lm.p1):
            self._lex_cum.append(self._lex_cum[-1] + (0.0 if i in lm.hidden else p))

    def _mass(self, prefix):
        """Unigram probability of the lexicon words that start with prefix (case-insensitive)."""
        low = prefix.lower()
        lo = bisect.bisect_left(self._lex, low)
        return self._lex_cum[bisect.bisect_left(self._lex, low + "\x7f", lo)] - self._lex_cum[lo]

    def _corrections(self, partial):
        """Typed letters -> [(intended letters, error weight)]: the typo_alts most likely single-error corrections
        that start a lexicon word."""
        if not self.typo_weight or len(partial) < self.typo_min:
            return []
        seen, scored = {partial}, []
        for q, _ in corrections(partial):
            if q not in seen:
                seen.add(q)
                if (mass := self._mass(q)) > 0:
                    scored.append((mass, q))
        scored.sort(reverse=True)
        return [(q, self.typo_weight) for _, q in scored[:self.typo_alts]]

    # --- Search ---

    @torch.inference_mode()
    def _search(self, context, partial, k):
        """Context and partial word (typeling.text.split_input) -> [(word, probability)], most probable first."""
        base = context.rstrip()
        lead = context[len(base):]
        if not base and not lead and self.prefix_space:
            lead = " "  # such a model starts every text with a space, so the first word is " word"
        self._context(base)
        # constraints: the lead + typed letters, and with typo repair lead + corrections, with log error weights
        fixes = self._corrections(partial)
        targets = [lead + partial] + [lead + q for q, _ in fixes]
        known = self._mass(partial) > 0 or self.user.prefixed(partial.lower(), 1)
        typed = self.unknown_weight if fixes and not known else 1.0
        tlogw = [math.log(typed)] + [math.log(w) for _, w in fixes]
        start, beam, strs, exact = len(lead), self.beam, self.strs, min(k, EXACT)
        words = {}   # lowercase word -> [probability, most probable form, its probability]
        z = 0.0      # probability of the constraints, weighted
        # (generated text, log probability incl. the constraint's weight, constraint, row of logp, token ids)
        hyps = [("", w, c, 0, ()) for c, w in enumerate(tlogw)]
        logp = self._logp0[None]
        try:
            for _ in range(max(map(len, targets)) + self.max_word_tokens):
                new = {}   # path (+ constraint until it is met) -> [log probability, parent, token id, text, constraint]

                def add(lp, h, t, s):
                    path, c = hyps[h][4] + (t,), hyps[h][2]
                    key = path if len(s) >= len(targets[c]) else (path, c)  # once met, constraints merge
                    e = new.get(key)
                    if e is None:
                        new[key] = [lp, h, t, s, c]
                    else:
                        e[0] = max(e[0], lp) + math.log1p(math.exp(-abs(e[0] - lp)))

                rows = [h for h, (g, _, c, _, _) in enumerate(hyps) if len(g) >= len(targets[c])]
                if rows:   # word phase, vectorized over rows
                    lp = logp[[hyps[h][3] for h in rows]]
                    p = lp.exp()
                    apos = torch.tensor([hyps[h][0].endswith("'") for h in rows], device=self.device)[:, None]
                    mask = torch.where(apos, self._letters, self._cont)
                    ends = (1.0 - (p * mask).sum(-1)).clamp_min(0.0).tolist()
                    top = torch.topk(lp.masked_fill(~mask, -math.inf), beam)
                    for r, h in enumerate(rows):
                        g, glp = hyps[h][:2]
                        word = g[start:].rstrip("'")
                        e = words.setdefault(word.lower(), [0.0, word, 0.0])
                        q = math.exp(glp) * ends[r]
                        e[0] += q
                        if q > e[2]:
                            e[1], e[2] = word, q
                        for v, t in zip(top.values[r].tolist(), top.indices[r].tolist()):
                            if v > -math.inf:
                                add(glp + v, h, t, g + strs[t])
                for h, (g, glp, c, row, _) in enumerate(hyps):
                    if len(g) >= len(targets[c]):
                        continue
                    rest = targets[c][len(g):]   # prefix phase
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
                                add(glp + v, h, t, s)
                    for i in range(1, len(rest)):
                        t = self._by_str.get(rest[:i])
                        if t is not None:
                            add(glp + lrow[t].item(), h, t, g + rest[:i])
                if not new:
                    break
                new = sorted(new.values(), reverse=True)[:beam]
                best = sorted((e[0] for e in words.values()), reverse=True)
                if len(best) >= exact and math.exp(new[0][0]) < best[exact - 1]:
                    break
                self._cache.reorder_cache(torch.tensor([hyps[e[1]][3] for e in new], device=self.device))
                out = self.model(input_ids=torch.tensor([[e[2]] for e in new], device=self.device),
                                 past_key_values=self._cache, use_cache=True)
                logp = torch.log_softmax(out.logits[:, -1].float(), -1)
                hyps = [(s, lp, c, r, hyps[h][4] + (t,)) for r, (lp, h, t, s, c) in enumerate(new)]
        finally:
            self._truncate(len(self._ids))
            self._cache.reorder_cache(torch.tensor([0], device=self.device))
        # hidden words are not suggested; P(word | not hidden)
        kept = {w: e for w, e in words.items() if w not in self._hidden and e[0] > 0}
        z = max(z - sum(e[0] for w, e in words.items() if w not in kept), sum(e[0] for e in kept.values()))
        u = self.user.weight()
        joint = {w: (1 - u) * e[0] for w, e in kept.items()}
        forms = {w: e[1] for w, e in kept.items()}
        if u:   # user model, mixed into the joint probabilities
            prev, pre, zu = self.ngram._context(context).prev, partial.lower(), 0.0
            for w in self.user.prefixed(pre, NGRAM_CANDIDATES):
                if w not in self._hidden:
                    p = self.user.prob(prev, w)
                    joint[w] = joint.get(w, 0.0) + u * p
                    forms[w] = self.user.forms.get(w) or forms.get(w, w)
                    zu += p
            z = (1 - u) * z + u * (zu if pre else 1.0)
        a = self.ngram_weight
        probs = {w: (1 - a) * p / z for w, p in joint.items()} if z > 0 else {}
        if a:
            for w, p in self.ngram.topk(context + partial, NGRAM_CANDIDATES):
                probs[w.lower()] = probs.get(w.lower(), 0.0) + a * p
                forms.setdefault(w.lower(), w)
        ranked = sorted(((p, forms[w]) for w, p in probs.items()), reverse=True)[:k]
        return [(w, p) for p, w in ranked]
