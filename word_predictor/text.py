"""
Tokenizer shared by training and runtime, so both see exactly the same tokens.

A normalized token is one of:
    word    lowercase, may contain inner apostrophes: i'm, don't, rock'n'roll
    <num>   a number
    <s>     sentence boundary: . ! ? ... and newlines
    ,       comma
    <p>     any other punctuation or symbol (quotes, brackets, colons, emoji, ...)
"""

import re

BOUNDARY, NUM, COMMA, PUNCT, UNK = "<s>", "<num>", ",", "<p>", "<unk>"
SPECIALS = (BOUNDARY, NUM, COMMA, PUNCT, UNK)

_TRANS = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2026": "...",
})
WORD = r"[A-Za-z]+(?:'[A-Za-z]+)*"
WORD_RE = re.compile(WORD)
_TOKEN = re.compile(rf"({WORD})|(\d+(?:[.,:]\d+)*)|([.!?]+|\n)|(,)|(\S)")
_PARTIAL = re.compile(r"[A-Za-z]+(?:'[A-Za-z]*)*$")
MAX_PARTIAL = 32


def normalize(text):
    """Replace curly quotes, dashes, non-breaking spaces and ellipses with their ASCII equivalents."""
    return text.translate(_TRANS)


def tokenize(text):
    """Normalized text -> [(token, original)]. The original (with its casing) is kept for words only;
    other tokens get None. Runs of <s> or <p> collapse into one."""
    out = []
    for m in _TOKEN.finditer(text):
        word, num, end, comma, _ = m.groups()
        if word:
            out.append((word.lower(), word))
            continue
        t = NUM if num else BOUNDARY if end else COMMA if comma else PUNCT
        if (t == BOUNDARY or t == PUNCT) and out and out[-1][0] == t:
            continue
        out.append((t, None))
    return out


def is_word(token):
    return token[0].isalpha()


def split_input(text):
    """Text before the cursor -> (context, partial word); an empty partial word means "predict the next word".
    Returns None when no suggestion should be offered: right after punctuation with no space yet,
    or when the partial "word" does not look like a word."""
    tail = text[-(MAX_PARTIAL + 1):]
    m = _PARTIAL.search(tail)
    if m:
        partial = m.group()
        if len(partial) > MAX_PARTIAL:
            return None
        return text[: len(text) - len(partial)], partial
    if not text or text[-1].isspace():
        return text, ""
    return None
