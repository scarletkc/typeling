"""
Binary model file (en.bin) reader/writer and interpolated Kneser-Ney trigram scoring.

File format (little-endian):
    b"WPRD"  u32 version  u32 metadata length  metadata (JSON)  u32 section count
    Each section: 4-byte name  1-byte type (B=u8 H=u16 I=u32 s=UTF-8 text)  3 padding bytes  u32 size  payload

Word ids follow the alphabetical order of the lowercased words, so the words sharing a prefix form one
contiguous id range. Special tokens come after all words.
Probabilities are quantized to one byte: q = round(-ln p / STEP), p = exp(-q * STEP).

Sections:
    VOCB  all words joined by "\\n"
    CASE  one "id\\tform" line per word whose usual form is not all lowercase (I'm, English)
    UNIG  unigram (continuation) probabilities, 1 byte per word
    BLCK  ids of offensive words, not suggested unless allow_blocked is set
    MISS  ids of likely misspellings (recieve, definately), never suggested but still used as context
    B2OF  bigrams: start offset of each context word's entries (V+1 values)
    B2GA  bigram backoff weights        B2ID/B2PR  entry word ids / probabilities
    T3KY  trigrams: context keys u*V+v (ascending)    T3OF  entry start offsets
    T3GA  trigram backoff weights       T3ID/T3PR  entry word ids / probabilities
    B2HK/B2HP, T3HK/T3HP  contexts with many entries: positions of their HEAD most probable entries
"""

import heapq
import json
import math
import struct
import sys
from array import array
from bisect import bisect_left
from itertools import accumulate

MAGIC = b"WPRD"
VERSION = 2
STEP = 0.08
HEAD = 16
HEAD_MIN = 256
DQ = tuple(math.exp(-q * STEP) for q in range(256))


def quantize(p):
    return max(0, min(255, round(-math.log(p) / STEP))) if p > 0 else 255


def write_model(path, meta, sections):
    """sections: {name: (type code, bytes)}; the bytes must already be little-endian."""
    blob = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(blob)) + blob + struct.pack("<I", len(sections)))
        for name, (code, data) in sections.items():
            f.write(name.encode("ascii") + code.encode("ascii") + b"\0\0\0" + struct.pack("<I", len(data)))
            f.write(data)


def read_model(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise ValueError(f"{path} is not a Typeling n-gram model file")
    version, n = struct.unpack_from("<II", data, 4)
    if version != VERSION:
        raise ValueError(f"unsupported model version {version} (expected {VERSION})")
    pos = 12 + n
    meta = json.loads(data[12:pos])
    (count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    sections = {}
    for _ in range(count):
        name, code = data[pos:pos + 4].decode("ascii"), chr(data[pos + 4])
        (size,) = struct.unpack_from("<I", data, pos + 8)
        pos += 12
        chunk = data[pos:pos + size]
        pos += size
        if code == "s":
            sections[name] = chunk.decode("utf-8")
        else:
            arr = array(code)
            arr.frombytes(chunk)
            if sys.byteorder == "big":
                arr.byteswap()
            sections[name] = arr
    return meta, sections


class Context:
    """Scoring state for a (u, v) context: bigram and trigram entry ranges and their backoff weights."""

    __slots__ = ("v", "ba", "bb", "g2", "tj", "ta", "tb", "g3")


class NgramModel:
    def __init__(self, path):
        meta, s = read_model(path)
        self.meta = meta
        self.words = s["VOCB"].split("\n")
        self.n_words = W = len(self.words)
        self.id = {w: i for i, w in enumerate(self.words)}
        for i, t in enumerate(meta["specials"]):
            self.id[t] = W + i
        self.V = W + len(meta["specials"])
        self.unk = self.id["<unk>"]
        self.p1 = [DQ[q] for q in s["UNIG"]]
        self.cum1 = list(accumulate(self.p1, initial=0.0))
        self.q1 = s["UNIG"]
        self.forms = {}
        for line in filter(None, s["CASE"].split("\n")):
            i, form = line.split("\t")
            self.forms[int(i)] = form
        self.blocked = set(s["BLCK"])
        self.misspelled = set(s["MISS"])
        self.hidden = self.blocked | self.misspelled
        self.b_off, self.b_gam, self.b_ids, self.b_pr = s["B2OF"], s["B2GA"], s["B2ID"], s["B2PR"]
        self.t_key, self.t_off, self.t_gam, self.t_ids, self.t_pr = s["T3KY"], s["T3OF"], s["T3GA"], s["T3ID"], s["T3PR"]
        self.b_head = self._heads(s["B2HK"], s["B2HP"])
        self.t_head = self._heads(s["T3HK"], s["T3HP"])
        self._uni_top = {}

    @staticmethod
    def _heads(keys, pos):
        return {k: pos[i * HEAD:(i + 1) * HEAD] for i, k in enumerate(keys)}

    def token_id(self, token):
        return self.id.get(token, self.unk)

    def prefix_range(self, prefix):
        """Id range [lo, hi) of the words starting with prefix."""
        if not prefix:
            return 0, self.n_words
        lo = bisect_left(self.words, prefix)
        return lo, bisect_left(self.words, prefix + "\x7f", lo)

    def form(self, i):
        return self.forms.get(i, self.words[i])

    def context(self, u, v):
        c = Context()
        c.v, c.ba, c.bb, c.g2 = v, self.b_off[v], self.b_off[v + 1], DQ[self.b_gam[v]]
        key = u * self.V + v
        j = bisect_left(self.t_key, key)
        if j < len(self.t_key) and self.t_key[j] == key:
            c.tj, c.ta, c.tb, c.g3 = j, self.t_off[j], self.t_off[j + 1], DQ[self.t_gam[j]]
        else:
            c.tj, c.ta, c.tb, c.g3 = -1, 0, 0, 1.0
        return c

    def unigram_context(self):
        """Context-free scoring state (unigram probabilities only), used for ablations."""
        c = Context()
        c.v, c.ba, c.bb, c.g2, c.tj, c.ta, c.tb, c.g3 = -1, 0, 0, 1.0, -1, 0, 0, 1.0
        return c

    def prob(self, c, w):
        """P(w | context) = p3(w) + g3 * (p2(w) + g2 * p1(w)), normalized over all tokens incl. punctuation."""
        p = self.p1[w]
        ids = self.b_ids
        j = bisect_left(ids, w, c.ba, c.bb)
        p = (DQ[self.b_pr[j]] if j < c.bb and ids[j] == w else 0.0) + c.g2 * p
        if c.tj >= 0:
            ids = self.t_ids
            j = bisect_left(ids, w, c.ta, c.tb)
            p = (DQ[self.t_pr[j]] if j < c.tb and ids[j] == w else 0.0) + c.g3 * p
        return p

    def range_mass(self, c, lo, hi):
        """Sum of P(w | context) over the word ids in [lo, hi)."""
        m = self.cum1[hi] - self.cum1[lo]
        a = bisect_left(self.b_ids, lo, c.ba, c.bb)
        b = bisect_left(self.b_ids, hi, a, c.bb)
        m = sum(map(DQ.__getitem__, self.b_pr[a:b])) + c.g2 * m
        if c.tj >= 0:
            a = bisect_left(self.t_ids, lo, c.ta, c.tb)
            b = bisect_left(self.t_ids, hi, a, c.tb)
            m = sum(map(DQ.__getitem__, self.t_pr[a:b])) + c.g3 * m
        return m

    def candidates(self, c, lo, hi, n):
        """Word ids in [lo, hi) that may rank near the top: the n best trigram entries, bigram entries and unigrams."""
        out = set(self._unigram_top(lo, hi, n))
        full = lo == 0 and hi == self.n_words
        for ids, pr, a0, b0, head in ((self.b_ids, self.b_pr, c.ba, c.bb, self.b_head.get(c.v)),
                                      (self.t_ids, self.t_pr, c.ta, c.tb, self.t_head.get(c.tj))):
            if b0 <= a0:
                continue
            a = bisect_left(ids, lo, a0, b0)
            b = bisect_left(ids, hi, a, b0)
            if b - a <= n:
                out.update(ids[a:b])
            elif full and head is not None:
                out.update(ids[i] for i in head[:n])
            else:
                out.update(ids[i] for i in heapq.nsmallest(n, range(a, b), key=pr.__getitem__))
        return out

    def _unigram_top(self, lo, hi, n):
        if hi - lo > 512:
            key = (lo, hi, n)
            top = self._uni_top.get(key)
            if top is None:
                if len(self._uni_top) > 4096:
                    self._uni_top.clear()
                top = self._uni_top[key] = self._scan_top(lo, hi, n)
            return top
        return self._scan_top(lo, hi, n)

    def _scan_top(self, lo, hi, n):
        q, hidden = self.q1, self.hidden
        ids = (i for i in range(lo, hi) if i not in hidden)
        return heapq.nsmallest(n, ids, key=q.__getitem__)
