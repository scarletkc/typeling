"""
Terminal autocomplete demo: completes the word being typed and predicts the next word after a space, as grey ghost
text and in a candidate bar below the line.

    python -m typeling                  # start typing; the grey text is the suggestion
    python -m typeling --no-learn       # do not learn or save your words
    python -m typeling --lm             # use Typeling-25M (pip install "typeling[lm]"); --lm DIR for another model
    python -m typeling --bench          # measure per-keystroke latency
    echo I wan | python -m typeling     # non-interactive: prints "I wan[t]"

Keys: Tab / Right accept the highlighted candidate (the first one, shown as ghost text when it completes the typed
letters; with the language model it may also correct them), Up / Down move the highlight, Esc hide, Backspace delete
a character, Ctrl+W delete a word, Enter new line, Ctrl+C quit.
Each line submitted with Enter is learned by the user model, which is stored only on this machine
(~/.typeling/user.json, shared by both models) and saved on exit.
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

from .predictor import Predictor
from .text import normalize, split_input

USER_FILE = Path.home() / ".typeling" / "user.json"

GREY, INVERT, RESET = "\x1b[90m", "\x1b[7m", "\x1b[0m"
PROMPT = "> "
ACCEPT_ADDS_SPACE = False  # append a space after an accepted suggestion
CANDIDATES = 5             # shown in the bar (the first 3 are the ones the benchmark measures)


class LineEditor:
    """Pure state machine: consumes keys and returns the strings to draw; never touches the terminal."""

    def __init__(self, completer, candidates=CANDIDATES):
        self.c = completer
        self.n = candidates
        self.text = ""
        self.hidden = False          # set by Esc; cleared by the next edit
        self.sel = 0                 # highlighted candidate
        self._cache = (None, "", [])  # (text, ghost, candidates): one inference per distinct text
        self.latencies = []

    def view(self):
        """-> (ghost text, [candidate words]) for the current text."""
        if self.hidden:
            return "", []
        if self._cache[0] != self.text:
            t0 = time.perf_counter()
            cands = [w for w, _ in self.c.topk(self.text, self.n)] if self.n else []
            g = self.c.suggest(self.text)
            self.latencies.append(time.perf_counter() - t0)
            self._cache = (self.text, g, cands)
        return self._cache[1], self._cache[2]

    def accept(self):
        """Replace the partly typed word with the highlighted candidate (or append the ghost text)."""
        g, cands = self.view()
        if self.sel == 0 and g:
            self.text += g
        elif cands:
            sp = split_input(normalize(self.text))
            partial = sp[1] if sp else ""
            self.text = self.text[:len(self.text) - len(partial)] + cands[self.sel]
        else:
            return
        self.text += " " if ACCEPT_ADDS_SPACE else ""

    def feed(self, key):
        """Handle one key; return "enter", "quit" or None."""
        if key is None:
            return None
        if key in ("UP", "DOWN"):
            n = len(self.view()[1])
            if n:
                self.sel = (self.sel + (1 if key == "DOWN" else -1)) % n
            return None
        if key in ("TAB", "RIGHT"):
            self.accept()
        elif key == "BS":
            self.text = self.text[:-1]
        elif key == "DELWORD":
            self.text = re.sub(r"\S*$", "", self.text.rstrip())
        elif key == "ESC":
            self.hidden = True
            return None
        elif key == "ENTER":
            return "enter"
        elif key == "QUIT":
            return "quit"
        elif len(key) == 1 and key.isprintable():
            self.text += key
        else:
            return None
        self.hidden = False
        self.sel = 0
        return None

    def render(self, width, show=True):
        """Two lines, the text with its ghost and the candidate bar; leaves the cursor after the text."""
        g, cands = self.view() if show else ("", [])
        avail = max(10, width - len(PROMPT) - 1)
        g = g[: avail - 1]
        room = avail - len(g)
        vis = self.text if len(self.text) <= room else self.text[-room:]
        s = "\r" + PROMPT + vis + "\x1b[K" + (f"{GREY}{g}{RESET}" if g else "")
        bar, used = [], len(PROMPT)
        for i, w in enumerate(cands):
            item = f"{i + 1} {w}"
            if used + len(item) + 2 > width - 1:
                break
            bar.append(f"{INVERT}{item}{RESET}" if i == self.sel else f"{GREY}{item}{RESET}")
            used += len(item) + 2
        return s + "\r\n\x1b[K" + " " * len(PROMPT) + "  ".join(bar) + f"\x1b[1A\r\x1b[{len(PROMPT) + len(vis)}C"


# Raw keyboard input, per platform.

if os.name == "nt":
    import ctypes
    import msvcrt

    _SPECIAL = {"M": "RIGHT", "K": "LEFT", "H": "UP", "P": "DOWN"}
    _KEYS = {"\x08": "BS", "\x7f": "DELWORD", "\x17": "DELWORD", "\r": "ENTER",
             "\t": "TAB", "\x1b": "ESC", "\x03": "QUIT", "\x04": "QUIT", "\x1a": "QUIT"}

    def _read_one():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return _SPECIAL.get(msvcrt.getwch())
        return _KEYS.get(ch, ch)

    def read_keys():
        """Block for one key, then drain any queued keys so fast typing triggers a single redraw."""
        keys = [_read_one()]
        while msvcrt.kbhit():
            keys.append(_read_one())
        return keys

    class raw_mode:
        def __enter__(self):
            # Enable ANSI escape sequences (colors, cursor movement) in this console.
            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(h, mode.value | 0x0004)
            return self

        def __exit__(self, *exc):
            return False

else:
    import termios
    import tty

    _ARROWS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
    _KEYS = {"\x7f": "BS", "\x08": "DELWORD", "\x17": "DELWORD", "\r": "ENTER",
             "\n": "ENTER", "\t": "TAB", "\x03": "QUIT", "\x04": "QUIT"}

    def read_keys():
        data = os.read(sys.stdin.fileno(), 1024).decode("utf-8", "ignore")
        keys, i = [], 0
        while i < len(data):
            if data[i] == "\x1b":
                if data[i + 1 : i + 2] in ("[", "O") and i + 2 < len(data):
                    keys.append(_ARROWS.get(data[i + 2]))
                    i += 3
                    continue
                keys.append("ESC")
            else:
                keys.append(_KEYS.get(data[i], data[i]))
            i += 1
        return keys

    class raw_mode:
        def __enter__(self):
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, *exc):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            return False


def _stats(latencies):
    if not latencies:
        return ""
    us = sorted(x * 1e6 for x in latencies)
    p99 = us[min(len(us) - 1, int(len(us) * 0.99))]
    return f"{len(us)} suggestions: mean {sum(us) / len(us):.0f} µs, p99 {p99:.0f} µs, max {us[-1]:.0f} µs"


def load(args, learn=False):
    """-> (predictor, description) for the command-line options."""
    user = USER_FILE if learn else None
    note = f"learning your words ({USER_FILE})" if learn else "learning off"
    if args.lm:
        from .lm import LMPredictor  # needs typeling[lm]
        if args.threads:
            import torch
            torch.set_num_threads(args.threads)
        return LMPredictor(args.lm, user, device=args.device), f"language model {args.lm} · {note}"
    return Predictor(user=user), note


def interactive(args):
    learn = not args.no_learn
    t0 = time.perf_counter()
    p, note = load(args, learn)
    ed = LineEditor(p, args.candidates)
    load_ms = (time.perf_counter() - t0) * 1000
    out = sys.stdout
    width = lambda: shutil.get_terminal_size().columns
    with raw_mode():
        out.write(f"{GREY}Model loaded in {load_ms:.0f} ms · {note} · Tab/→ accept · ↑/↓ choose · Esc hide · "
                  f"Ctrl+W delete word · Enter new line · Ctrl+C quit{RESET}\r\n")
        out.write(ed.render(width()))
        out.flush()
        try:
            while True:
                for k in read_keys():
                    r = ed.feed(k)
                    if r == "quit":
                        raise KeyboardInterrupt
                    if r == "enter":
                        out.write(ed.render(width(), show=False) + "\r\n")
                        if learn and ed.text.strip():
                            p.learn(ed.text)
                        ed.text = ""
                out.write(ed.render(width()))
                out.flush()
        except KeyboardInterrupt:
            out.write(ed.render(width(), show=False) + "\r\n")
    if learn:
        p.save()
    print(_stats(ed.latencies))


def pipe_mode(args):
    c, _ = load(args)
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        g = c.suggest(line)
        print(f"{line}[{g}]" if g else line)


def bench(args):
    t0 = time.perf_counter()
    c, note = load(args)
    print(f"Model loaded in {(time.perf_counter() - t0) * 1000:.0f} ms ({note})")
    text = ("I want to know whether this little model can keep up with really fast "
            "typing because everyone expects instant suggestions while writing ") * 30
    ed = LineEditor(c, args.candidates)
    for ch in text:              # type one key at a time
        ed.feed(ch)
        ed.view()
    for _ in range(200):         # then backspace one key at a time
        ed.feed("BS")
        ed.view()
    print(_stats(ed.latencies))
    for s in ["I wan", "th", "Hel", "WOR", "encyc", "(beaut", "I don", "xylo", "Thank you so ", "how are ",
              "See you tomm"]:
        g = c.suggest(s)
        cands = ", ".join(w for w, _ in c.topk(s, args.candidates))
        print(f"  {s!r:>15} → {s}{GREY}{g}{RESET}" + ("" if g else "   (no suggestion)") + f"   [{cands}]")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="typeling", description="Terminal autocomplete demo")
    ap.add_argument("--no-learn", action="store_true", help="do not learn or save your words")
    ap.add_argument("--lm", metavar="MODEL", nargs="?", const="scarletkc/Typeling-25M",
                    help='use the language model (pip install "typeling[lm]"): Typeling-25M from Hugging Face, or '
                    "another repo id or directory")
    ap.add_argument("--device", default=None, help="language model device: cpu or cuda (default: cuda if available)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for the language model")
    ap.add_argument("--candidates", type=int, default=CANDIDATES, help=f"candidates in the bar (default {CANDIDATES})")
    ap.add_argument("--bench", action="store_true", help="measure per-keystroke latency")
    args = ap.parse_args()
    if args.bench:
        bench(args)
    elif not sys.stdin.isatty():
        pipe_mode(args)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
