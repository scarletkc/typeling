"""Fonts, colors and output paths shared by the blog image scripts (make_demo.py, make_images.py)."""

from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"

# first existing file wins: Windows, macOS, Linux
FONTS = {
    "sans": ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Helvetica.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "semibold": ["C:/Windows/Fonts/seguisb.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono": ["C:/Windows/Fonts/CascadiaMono.ttf", "C:/Windows/Fonts/consola.ttf", "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}

INK = "#1F2328"       # text on light backgrounds
MUTED = "#656D76"
ACCENT = "#6D4AFF"    # Typeling
NGRAM = "#8C959F"     # n-gram, 23 MB
NGRAM_SMALL = "#C9D1D9"
DARK_BG = "#0D1117"   # terminal and cover
DARK_PANEL = "#161B22"
DARK_TEXT = "#E6EDF3"
DARK_MUTED = "#6E7681"


def font_path(kind):
    for path in FONTS[kind]:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"no {kind} font found; add one to common.FONTS")


def font(kind, size):
    return ImageFont.truetype(font_path(kind), size)
