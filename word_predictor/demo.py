"""
Terminal ghost-text autocomplete demo: completes the word being typed and predicts the next word after a space.

    python -m word_predictor                  # start typing; the grey text is the suggestion
    python -m word_predictor --no-learn       # do not learn or save your words
    python -m word_predictor --bench          # measure per-keystroke latency
    echo I wan | python -m word_predictor     # non-interactive: prints "I wan[t]"

Keys: Tab / Right accept, Esc hide, Backspace delete a character, Ctrl+W delete a word, Enter new line, Ctrl+C quit.
Each line submitted with Enter is learned by the user model, which is stored only on this machine
(~/.word_predictor/user.json) and saved on exit.
"""

import os
import re
import shutil
import sys
import time
from pathlib import Path

from .predictor import Predictor

USER_FILE = Path.home() / ".word_predictor" / "user.json"

GREY, RESET = "\x1b[90m", "\x1b[0m"
PROMPT = "> "
ACCEPT_ADDS_SPACE = False  # append a space after an accepted suggestion


class LineEditor:
    """Pure state machine: consumes keys and returns the string to draw; never touches the terminal."""

    def __init__(self, completer):
        self.c = completer
        self.text = ""
        self.hidden = False          # set by Esc; cleared by the next edit
        self._cache = (None, "")     # (text, ghost): one inference per distinct text
        self.latencies = []

    def ghost(self):
        if self.hidden:
            return ""
        if self._cache[0] != self.text:
            t0 = time.perf_counter()
            g = self.c.suggest(self.text)
            self.latencies.append(time.perf_counter() - t0)
            self._cache = (self.text, g)
        return self._cache[1]

    def feed(self, key):
        """Handle one key; return "enter", "quit" or None."""
        if key is None:
            return None
        if key in ("TAB", "RIGHT"):
            g = self.ghost()
            if g:
                self.text += g + (" " if ACCEPT_ADDS_SPACE else "")
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
        return None

    def render(self, width, show_ghost=True):
        g = self.ghost() if show_ghost else ""
        avail = max(10, width - len(PROMPT) - 1)
        g = g[: avail - 1]
        room = avail - len(g)
        vis = self.text if len(self.text) <= room else self.text[-room:]
        s = "\r" + PROMPT + vis + "\x1b[K"
        if g:
            s += f"{GREY}{g}{RESET}\x1b[{len(g)}D"
        return s


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

    _KEYS = {"\x7f": "BS", "\x08": "DELWORD", "\x17": "DELWORD", "\r": "ENTER",
             "\n": "ENTER", "\t": "TAB", "\x03": "QUIT", "\x04": "QUIT"}

    def read_keys():
        data = os.read(sys.stdin.fileno(), 1024).decode("utf-8", "ignore")
        keys, i = [], 0
        while i < len(data):
            if data[i] == "\x1b":
                if data[i + 1 : i + 2] in ("[", "O") and i + 2 < len(data):
                    keys.append("RIGHT" if data[i + 2] == "C" else None)
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


def interactive(learn):
    t0 = time.perf_counter()
    p = Predictor(user=USER_FILE if learn else None)
    ed = LineEditor(p)
    load_ms = (time.perf_counter() - t0) * 1000
    out = sys.stdout
    width = lambda: shutil.get_terminal_size().columns
    note = f"learning your words ({USER_FILE})" if learn else "learning off"
    with raw_mode():
        out.write(f"{GREY}Model loaded in {load_ms:.0f} ms · {note} · Tab/→ accept · Esc hide · "
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
                        out.write(ed.render(width(), show_ghost=False) + "\r\n")
                        if learn and ed.text.strip():
                            p.learn(ed.text)
                        ed.text = ""
                out.write(ed.render(width()))
                out.flush()
        except KeyboardInterrupt:
            out.write(ed.render(width(), show_ghost=False) + "\r\n")
    if learn:
        p.save()
    print(_stats(ed.latencies))


def pipe_mode():
    c = Predictor()
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        g = c.suggest(line)
        print(f"{line}[{g}]" if g else line)


def bench():
    t0 = time.perf_counter()
    c = Predictor()
    print(f"Model loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    text = ("I want to know whether this little model can keep up with really fast "
            "typing because everyone expects instant suggestions while writing ") * 30
    ed = LineEditor(c)
    for ch in text:              # type one key at a time
        ed.feed(ch)
        ed.ghost()
    for _ in range(200):         # then backspace one key at a time
        ed.feed("BS")
        ed.ghost()
    print(_stats(ed.latencies))
    for s in ["I wan", "th", "Hel", "WOR", "encyc", "(beaut", "I don", "xylo", "Thank you so ", "how are "]:
        g = c.suggest(s)
        print(f"  {s!r:>10} → {s}{GREY}{g}{RESET}" + ("" if g else "   (no suggestion)"))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if "--bench" in sys.argv[1:]:
        bench()
    elif not sys.stdin.isatty():
        pipe_mode()
    else:
        interactive(learn="--no-learn" not in sys.argv[1:])


if __name__ == "__main__":
    main()
