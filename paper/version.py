"""Writes paper/version.tex with the package version from pyproject.toml, so the technical report carries the version
of the release it is attached to (.github/workflows/publish.yml).

    uv run --no-project --python 3.13 paper/version.py
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
(ROOT / "paper" / "version.tex").write_text(f"\\newcommand{{\\typelingversion}}{{{version}}}\n", encoding="utf-8")
print(f"paper/version.tex: version {version}")
