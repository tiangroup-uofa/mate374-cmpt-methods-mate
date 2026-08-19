#!/usr/bin/env python3
"""Build a direct-source editable molab URL for one marimo notebook."""

from __future__ import annotations

import sys
from pathlib import Path

from lzstring import LZString


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: marimo_iframe_url.py NOTEBOOK.py")

    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    compressed = LZString().compressToEncodedURIComponent(source)
    print(
        "https://marimo.app/"
        "?embed=true&mode=edit&show-chrome=false"
        f"#code/{compressed}"
    )


if __name__ == "__main__":
    main()
