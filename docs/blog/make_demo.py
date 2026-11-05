# /// script
# requires-python = ">=3.13"
# dependencies = ["torch>=2.7", "transformers>=5", "huggingface_hub", "numpy", "pillow>=11"]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""
Renders the terminal demo (typeling --lm with a bar of 3 candidates) as an animated GIF and stills for the blog post.
Every frame shows what typeling.demo.LineEditor displays after that key, from the real suggestions of Typeling-25M.

    uv run docs/blog/make_demo.py                                  # data/release/Typeling-25M if present, else the Hub
    uv run docs/blog/make_demo.py --model ScarletKc/Typeling-25M

Writes docs/blog/images/demo.gif, demo-next-word.png, demo-completion.png and demo-typo.png.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from common import DARK_BG, DARK_MUTED, DARK_PANEL, DARK_TEXT, IMAGES, font

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from typeling.demo import LineEditor  # noqa: E402
from typeling.lm import LMPredictor  # noqa: E402
from typeling.text import normalize, split_input  # noqa: E402

W, H = 1200, 360
PAD = 40
LABEL = "#B49CFF"
TYPED_MS, HOLD_MS, TAB_MS, END_MS = 110, 1000, 600, 3200
# keys: text typed one character at a time, "TAB" accepts, "HOLD" pauses on the current frame
SCRIPT = ["Thank you so ", "HOLD", "TAB", "! See y", "HOLD", "TAB", " ", "HOLD", "tomm", "HOLD", "TAB", "!"]

MONO, MONO_BAR, SANS, SANS_SMALL = font("mono", 34), font("mono", 26), font("sans", 22), font("sans", 20)


def label(text, ghost, cands, tab):
    """What the frame demonstrates."""
    if tab:
        return "Tab: accept"
    sp = split_input(normalize(text))
    partial = sp[1] if sp else ""
    if ghost:
        return "next-word prediction" if not partial else "word completion"
    if cands and partial and not cands[0].lower().startswith(partial.lower()):
        return "typo repair: the correction is offered in the bar"
    return ""


def frame(text, ghost, cands, tab=False):
    im = Image.new("RGB", (W, H), DARK_BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 44], fill=DARK_PANEL)
    for i, c in enumerate(("#FF5F56", "#FFBD2E", "#27C93F")):
        d.ellipse([20 + i * 26, 15, 34 + i * 26, 29], fill=c)
    d.text((W / 2, 22), "typeling --lm", font=SANS_SMALL, fill=DARK_MUTED, anchor="mm")
    d.text((PAD, 74), "Typeling-25M  ·  Tab accept  ·  ↑/↓ choose  ·  runs offline", font=SANS_SMALL, fill=DARK_MUTED)
    y = 150
    x = PAD
    d.text((x, y), "> ", font=MONO, fill=DARK_MUTED, anchor="ls")
    x += d.textlength("> ", font=MONO)
    d.text((x, y), text, font=MONO, fill=DARK_TEXT, anchor="ls")
    x += d.textlength(text, font=MONO)
    d.rectangle([x + 1, y - 30, x + 4, y + 6], fill=DARK_TEXT)   # caret
    if ghost:
        d.text((x + 6, y), ghost, font=MONO, fill=DARK_MUTED, anchor="ls")
    if tab:
        d.rounded_rectangle([W - PAD - 86, y - 34, W - PAD, y + 8], radius=8, fill="#30363D")
        d.text((W - PAD - 43, y - 13), "Tab", font=SANS, fill=DARK_TEXT, anchor="mm")
    x, y = PAD + d.textlength("> ", font=MONO), 222
    for i, w in enumerate(cands):
        item = f"{i + 1} {w}"
        width = d.textlength(item, font=MONO_BAR)
        if i == 0:
            d.rounded_rectangle([x - 8, y - 26, x + width + 8, y + 10], radius=6, fill=DARK_TEXT)
            d.text((x, y), item, font=MONO_BAR, fill=DARK_BG, anchor="ls")
        else:
            d.text((x, y), item, font=MONO_BAR, fill=DARK_MUTED, anchor="ls")
        x += width + 34
    note = label(text, ghost, cands, tab)
    if note:
        d.text((PAD, H - 44), note, font=SANS, fill=LABEL, anchor="ls")
    return im


def still(p, text, name):
    ed = LineEditor(p, 3)
    ed.text = text
    frame(text, *ed.view()).save(IMAGES / name)


def main():
    ap = argparse.ArgumentParser(description="Render the terminal demo for the blog post")
    default = ROOT / "data" / "release" / "Typeling-25M"
    ap.add_argument("--model", default=str(default) if default.exists() else "ScarletKc/Typeling-25M")
    args = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    p = LMPredictor(args.model, device="cpu")
    ed = LineEditor(p, 3)
    frames, durations = [frame("", *ed.view())], [600]
    for key in SCRIPT:
        if key == "HOLD":
            durations[-1] += HOLD_MS
        elif key == "TAB":
            ed.feed("TAB")
            frames.append(frame(ed.text, *ed.view(), tab=True))
            durations.append(TAB_MS)
        else:
            for ch in key:
                ed.feed(ch)
                frames.append(frame(ed.text, *ed.view()))
                durations.append(TYPED_MS)
    frames.append(frame(ed.text, *ed.view()))
    durations.append(END_MS)
    sheet = Image.new("RGB", (W, H * len(frames)))
    for i, f in enumerate(frames):
        sheet.paste(f, (0, i * H))
    palette = sheet.quantize(colors=96, dither=Image.Dither.NONE)   # one palette for all frames: no flicker
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(IMAGES / "demo.gif", save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
    still(p, "Thank you so ", "demo-next-word.png")
    still(p, "Let me kn", "demo-completion.png")
    still(p, "Thank you so much! See you tomm", "demo-typo.png")
    print(f"{len(frames)} frames, {sum(durations) / 1000:.1f} s -> {IMAGES / 'demo.gif'}")


if __name__ == "__main__":
    main()
