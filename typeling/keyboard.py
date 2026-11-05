"""
QWERTY keyboard error model: the strings that a single typing error turns into the typed letters.

    from typeling.keyboard import corrections
    {q for q, kind in corrections("tomm")}    # contains "tom" (extra key), "tpmm" (adjacent key), ...

Used by the language model decoder's typo repair (typeling.lm) and by the typo benchmark (lm/typos.py).
"""

ROWS, OFFSETS = ("qwertyuiop", "asdfghjkl", "zxcvbnm"), (0.0, 0.25, 0.75)
ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def _neighbors():
    pos = {c: (r, i + OFFSETS[r]) for r, row in enumerate(ROWS) for i, c in enumerate(row)}
    return {c: "".join(d for d, (r2, x2) in pos.items() if d != c and (
        (r2 == r and abs(x2 - x) <= 1) or (abs(r2 - r) == 1 and abs(x2 - x) < 1))) for c, (r, x) in pos.items()}


NEIGHBORS = _neighbors()  # g -> tyfhvb


def corrections(t):
    """Typed letters -> (intended letters, kind) for each single error that turns the intended letters into t:
    sub   an adjacent key was hit instead
    del   an extra key was hit, doubling or next to a neighboring letter
    ins   a letter or apostrophe was left out (not at the end, where it is simply not typed yet)
    swap  two neighboring letters were swapped
    case  the first letter was not capitalized
    Results may repeat and need not be words."""
    low = t.lower()
    for i, c in enumerate(t):
        nb = NEIGHBORS.get(low[i], "")
        for d in nb:
            yield t[:i] + (d.upper() if c.isupper() else d) + t[i + 1:], "sub"
        if len(t) > 1 and any(x == low[i] or x in nb for x in low[max(0, i - 1):i] + low[i + 1:i + 2]):
            yield t[:i] + t[i + 1:], "del"
        for d in ALPHABET[:-1] if i == 0 else ALPHABET:
            yield t[:i] + d + t[i:], "ins"
        if i + 1 < len(t) and low[i] != low[i + 1]:
            yield t[:i] + t[i + 1] + c + t[i + 2:], "swap"
    if t[:1].islower():
        yield t[0].upper() + t[1:], "case"
