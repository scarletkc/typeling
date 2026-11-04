# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
终端里的灰色自动补全（基于 complete_kernel.Completer）。

    python autocomplete.py                  # 开始打字，灰色部分是补全
    python autocomplete.py --bench          # 测每次按键的补全耗时
    echo I wan | python autocomplete.py     # 非交互：打印 "I wan[t]"

按键：Tab / → 接受补全 · Esc 隐藏补全 · Backspace 删字 · Ctrl+W 删词 · Enter 换行 · Ctrl+C 退出
"""

import os
import re
import shutil
import sys
import time

from complete_kernel import Completer

GREY, RESET = "\x1b[90m", "\x1b[0m"
PROMPT = "> "
ACCEPT_ADDS_SPACE = False  # 接受补全后是否顺手补一个空格


class LineEditor:
    """纯状态机：吃按键、产出要画的字符串，不直接碰终端。"""

    def __init__(self, completer):
        self.c = completer
        self.text = ""
        self.hidden = False          # Esc 之后隐藏补全，直到下一次输入
        self._cache = (None, "")     # (text, ghost)：同一行文字只推理一次
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
        """处理一个按键；返回 "enter" / "quit" / None。"""
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


# ───────────────────────── 终端读键 ─────────────────────────

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
        """阻塞读一个键，再把已经排队的键一起读走（打字快时合并成一次重绘）。"""
        keys = [_read_one()]
        while msvcrt.kbhit():
            keys.append(_read_one())
        return keys

    class raw_mode:
        def __enter__(self):
            # 打开本控制台的 ANSI 转义支持（灰色、光标移动）
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


# ───────────────────────── 三种模式 ─────────────────────────


def _stats(latencies):
    if not latencies:
        return ""
    us = sorted(x * 1e6 for x in latencies)
    p99 = us[min(len(us) - 1, int(len(us) * 0.99))]
    return f"补全 {len(us)} 次：平均 {sum(us) / len(us):.0f} µs，p99 {p99:.0f} µs，最慢 {us[-1]:.0f} µs"


def interactive():
    t0 = time.perf_counter()
    ed = LineEditor(Completer())
    load_ms = (time.perf_counter() - t0) * 1000
    out = sys.stdout
    width = lambda: shutil.get_terminal_size().columns
    with raw_mode():
        out.write(f"{GREY}权重加载 {load_ms:.0f} ms · Tab/→ 接受 · Esc 隐藏 · Ctrl+W 删词 · "
                  f"Enter 换行 · Ctrl+C 退出{RESET}\r\n")
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
                        ed.text = ""
                out.write(ed.render(width()))
                out.flush()
        except KeyboardInterrupt:
            out.write(ed.render(width(), show_ghost=False) + "\r\n")
    print(_stats(ed.latencies))


def pipe_mode():
    c = Completer()
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        g = c.suggest(line)
        print(f"{line}[{g}]" if g else line)


def bench():
    t0 = time.perf_counter()
    c = Completer()
    print(f"权重加载 {(time.perf_counter() - t0) * 1000:.0f} ms")
    text = ("I want to know whether this little model can keep up with really fast "
            "typing because everyone expects instant suggestions while writing ") * 30
    ed = LineEditor(c)
    for ch in text:              # 逐键输入
        ed.feed(ch)
        ed.ghost()
    for _ in range(200):         # 再逐键删回去
        ed.feed("BS")
        ed.ghost()
    print(_stats(ed.latencies))
    for s in ["I wan", "th", "Hel", "WOR", "encyc", "(beaut", "don't", "xylo"]:
        g = c.suggest(s)
        print(f"  {s!r:>10} → {s}{GREY}{g}{RESET}" + ("" if g else "   （无补全）"))


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
        interactive()


if __name__ == "__main__":
    main()
