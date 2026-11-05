# /// script
# requires-python = ">=3.13"
# dependencies = ["matplotlib>=3.9", "pillow>=11"]
# ///
"""
Renders the cover, diagrams and charts of the blog post into docs/blog/images/. Every number is from
docs/model-plan.md (and the technical report); the demo GIF and stills come from make_demo.py.

    uv run docs/blog/make_images.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from common import (ACCENT, DARK_BG, DARK_MUTED, DARK_TEXT, IMAGES, INK, MUTED, NGRAM, NGRAM_SMALL,  # noqa: E402
                    font, font_path)

# Test split, first 2,000 words per source (model-plan section 15)
DOMAINS = ["Chat", "Casual\n(IRC)", "Writing\n(FineWeb)"]
KSR = {"n-gram, 5 MB": [31.6, 22.7, 22.3], "n-gram, 23 MB": [34.1, 24.4, 23.9], "Typeling-25M": [43.1, 29.8, 32.4]}
KSR3 = {"n-gram, 5 MB": [48.8, 38.9, 40.6], "n-gram, 23 MB": [50.6, 40.4, 41.5], "Typeling-25M": [58.0, 46.8, 49.9]}
COLORS = [NGRAM_SMALL, NGRAM, ACCENT]
# Dev split, 2,000 words per source, early decoder (sections 12 and 13)
STAGES = ["n-gram\n(23 MB)", "Base model,\nzero-shot", "+20M tokens\n(5.6 min)", "+150M tokens\n(58 min)"]
GAIN = {"Chat": [31.9, 33.5, 40.3, 41.2], "Casual (IRC)": [24.8, 22.8, 27.7, 29.6]}
# "Let me know " (section 12: Supra2-Medium-Base raw next token; technical report appendix B: Typeling-25M decoder)
RAW = [('"!"', 0.37), ("invalid byte", 0.19), ('"!!"', 0.08)]
DECODED = [("if", 0.73), ("how", 0.12), ("what", 0.06)]


def setup_matplotlib():
    for kind in ("sans", "bold"):
        font_manager.fontManager.addfont(font_path(kind))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_path("sans")).get_name(),
        "font.size": 13, "axes.edgecolor": "#D0D7DE", "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 15, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def bars(ax, groups, series, colors, fmt="{:.1f}"):
    n = len(series)
    width = 0.8 / n
    for i, ((name, values), color) in enumerate(zip(series.items(), colors)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        rects = ax.bar(xs, values, width * 0.92, label=name, color=color)
        for r, v in zip(rects, values):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.6, fmt.format(v), ha="center", va="bottom", fontsize=10.5,
                    color=INK, fontweight="bold" if color == ACCENT else "normal")
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#EAEEF2")
    ax.set_axisbelow(True)


def results():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, data, title in zip(axes, (KSR, KSR3), ("Ghost text (KSR)", "Bar of 3 candidates (KSR@3)")):
        bars(ax, DOMAINS, data, COLORS)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 66)
    axes[0].set_ylabel("Keystrokes saved (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=11.5, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.01, -0.16, "Held-out test split, first 2,000 words per source. Higher is better.", color=MUTED,
             fontsize=10.5)
    fig.savefig(IMAGES / "results.png")
    plt.close(fig)


def gain():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars(ax, STAGES, GAIN, [ACCENT, "#B49CFF"])
    ax.set_ylim(0, 56)
    ax.set_ylabel("Keystrokes saved, ghost text (%)")
    ax.set_title("Where the gain over the n-gram model comes from", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.annotate("", xy=(3 - 0.2, 47.5), xytext=(1 - 0.2, 47.5),
                arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.6})
    ax.text(2 - 0.2, 48.6, "+7.7 points on chat from continued pretraining", ha="center", va="bottom",
            color=ACCENT, fontsize=11, fontweight="bold")
    fig.text(0.01, -0.06, "Development split, 2,000 words per source, through the same early decoder "
             "(no typo repair or n-gram fusion).", color=MUTED, fontsize=10.5)
    fig.savefig(IMAGES / "gain.png")
    plt.close(fig)


def trailing_space():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"wspace": 0.5})
    panels = ((RAW, NGRAM, "Raw model, next token after “Let me know ”"),
              (DECODED, ACCENT, "Typeling decoder, next word"))
    for ax, (items, color, title) in zip(axes, panels):
        labels, values = [k for k, _ in items][::-1], [v for _, v in items][::-1]
        ax.barh(labels, values, color=color, height=0.6)
        for i, v in enumerate(values):
            ax.text(v + 0.015, i, f"{v:.0%}", va="center", fontsize=11, color=INK)
        ax.set_xlim(0, 0.85)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=INK)
        ax.set_title(title, loc="left")
    fig.text(0.01, -0.08, "Left: Supra2-Medium-Base before any training; byte-level BPE attaches the space to the next "
             "word, so text ending in a space looks unusual.\nRight: Typeling-25M through the word decoder, which "
             "cuts the space off and constrains the next token to start with it.", color=MUTED, fontsize=10.5)
    fig.savefig(IMAGES / "trailing-space.png")
    plt.close(fig)


def chip(d, x, y, text, fnt, fill, color, outline=None, pad=14, strike=False):
    """Rounded label with its left edge at x, vertically centered on y; returns its right edge."""
    w = d.textlength(text, font=fnt)
    d.rounded_rectangle([x, y - 30, x + w + 2 * pad, y + 30], radius=10, fill=fill, outline=outline, width=3)
    d.text((x + pad, y), text, font=fnt, fill=color, anchor="lm")
    if strike:
        d.line([x + pad - 4, y, x + pad + w + 4, y], fill=color, width=3)
    return x + w + 2 * pad


def arrow(d, x0, x1, y, color=MUTED):
    d.line([x0, y, x1 - 10, y], fill=color, width=3)
    d.polygon([(x1, y), (x1 - 14, y - 8), (x1 - 14, y + 8)], fill=color)


def cover():
    W, H = 2000, 800          # 5:2, everything inside the central safe area
    im = Image.new("RGB", (W, H), DARK_BG)
    d = ImageDraw.Draw(im)
    d.text((W / 2, 190), "Typeling", font=font("bold", 132), fill=DARK_TEXT, anchor="mm")
    d.text((W / 2, 335), "A 25M-parameter language model for offline autocomplete", font=font("sans", 46),
           fill="#9DA7B3", anchor="mm")
    mono, typed, ghost = font("mono", 60), "> Let me kn", "ow"
    x = (W - d.textlength(typed + ghost, font=mono) - 10) / 2
    y = 478
    d.text((x, y), typed, font=mono, fill=DARK_TEXT, anchor="ls")
    x += d.textlength(typed, font=mono)
    d.rectangle([x + 2, y - 52, x + 6, y + 8], fill=DARK_TEXT)
    d.text((x + 10, y), ghost, font=mono, fill=DARK_MUTED, anchor="ls")
    bar = font("mono", 40)
    items = ["1 know", "2 knock", "3 know's"]
    x = (W - sum(d.textlength(t, font=bar) for t in items) - 2 * 48) / 2
    for i, t in enumerate(items):
        w = d.textlength(t, font=bar)
        if i == 0:
            d.rounded_rectangle([x - 12, 520, x + w + 12, 576], radius=8, fill=DARK_TEXT)
        d.text((x, 548), t, font=bar, fill=DARK_BG if i == 0 else DARK_MUTED, anchor="lm")
        x += w + 48
    d.text((W / 2, 668), "43% of keystrokes saved on chat (n-gram: 34%)  ·  27.6 MB  ·  trained in 58 minutes on a "
           "laptop", font=font("sans", 34), fill="#B49CFF", anchor="mm")
    im.save(IMAGES / "cover.png")


def ksr():
    W, H = 1600, 640
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((80, 80), "How keystroke savings are counted", font=font("bold", 48), fill=INK, anchor="ls")
    mono, small = font("mono", 54), font("sans", 26)
    x, y = 80, 250
    d.text((x, y), "…See you ", font=mono, fill=MUTED, anchor="lm")
    x += d.textlength("…See you ", font=mono) + 6
    box, gap = 70, 10
    start = x
    for i, ch in enumerate("tomorrow"):
        typed = i < 2
        d.rounded_rectangle([x, y - 46, x + box, y + 46], radius=10, fill=INK if typed else "#EEF0F3")
        d.text((x + box / 2, y), ch, font=mono, fill="white" if typed else MUTED, anchor="mm")
        x += box + gap
    d.text((start + box + gap / 2, y + 80), "typed: 2 keys", font=small, fill=INK, anchor="mm")
    d.text((start + 2 * (box + gap) + (6 * box + 5 * gap) / 2, y + 80), "shown as ghost text", font=small, fill=MUTED,
           anchor="mm")
    x += 26
    d.rounded_rectangle([x, y - 46, x + 140, y + 46], radius=12, outline=ACCENT, width=4)
    d.text((x + 70, y), "Tab", font=font("semibold", 40), fill=ACCENT, anchor="mm")
    d.text((x + 70, y + 80), "accept: 1 key", font=small, fill=ACCENT, anchor="mm")
    d.text((80, 440), "8 letters, 3 keystrokes: 5 saved (62.5%)", font=font("semibold", 38), fill=INK, anchor="ls")
    body = font("sans", 30)
    d.text((80, 510), "KSR = 1 − keystrokes ÷ letters, over every word of held-out text.", font=body, fill=MUTED,
           anchor="ls")
    d.text((80, 560), "KSR@3: the same, but the word can also be picked from a bar of 3 candidates.", font=body,
           fill=MUTED, anchor="ls")
    im.save(IMAGES / "ksr.png")


def decoder():
    W, H = 1600, 960
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    tok, head, body = font("mono", 34), font("semibold", 34), font("sans", 28)
    d.text((80, 80), "Turning tokens into words", font=font("bold", 48), fill=INK, anchor="ls")
    y = 170
    d.text((80, y), "Text before the cursor:", font=body, fill=MUTED, anchor="lm")
    x = chip(d, 420, y, '"Let me"', tok, "#EEF0F3", INK) + 16
    x = chip(d, x, y, '" kn"', tok, "white", ACCENT, outline=ACCENT) + 24
    d.text((x, y), "context (cached)  +  typed letters = constraint", font=body, fill=MUTED, anchor="lm")

    y = 300
    d.text((80, y), "1  Prefix phase: only tokens that agree with \" kn\"", font=head, fill=INK, anchor="lm")
    x = 120
    for t, ok in (('" k"', True), ('" kn"', True), ('" know"', True), ('" ka"', False), ('"!"', False)):
        x = chip(d, x, y + 80, t, tok, "#F1EDFF" if ok else "#FFF1F0", ACCENT if ok else "#CF222E",
                 strike=not ok) + 20
    d.text((x + 10, y + 80), "allowed / rejected", font=body, fill=MUTED, anchor="lm")

    y = 500
    d.text((80, y), "2  Word phase: add letter tokens until the model ends the word", font=head, fill=INK,
           anchor="lm")
    for row, (pieces, word) in enumerate(((['" know"'], "know"), (['" know"', '"ing"'], "knowing"),
                                          (['" know"', "\"'s\""], "know's"))):
        yy = y + 80 + row * 78
        x = 120
        for i, t in enumerate(pieces):
            if i:
                x = chip(d, x + 8, yy, "+", tok, "white", MUTED, pad=4) + 8
            x = chip(d, x, yy, t, tok, "#F1EDFF", ACCENT)
        arrow(d, x + 18, 600, yy)
        x = chip(d, 614, yy, "word ends", font("sans", 26), "#EEF0F3", MUTED)
        arrow(d, x + 18, x + 90, yy)
        d.text((x + 108, yy), word, font=font("semibold", 36), fill=INK, anchor="lm")

    y = 860
    d.text((80, y), "P(word) = Σ over its token paths  P(path) × P(word ends)  ÷  P(\" kn\")", font=head, fill=INK,
           anchor="lm")
    d.text((80, y + 56), "The top 3 words fill the candidate bar; the first becomes ghost text if it extends the typed "
           "letters.", font=body, fill=MUTED, anchor="lm")
    im.save(IMAGES / "decoder.png")


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    for make in (results, gain, trailing_space, cover, ksr, decoder):
        make()
        print("wrote", make.__name__)


if __name__ == "__main__":
    main()
