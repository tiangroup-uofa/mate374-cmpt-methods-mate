#!/usr/bin/env python3
"""Build a direct-source editable marimo.app URL for one marimo notebook.

With --share, print a full-editor link instead of an embeddable one.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lzstring import LZString


def main() -> None:
    args = sys.argv[1:]
    share = "--share" in args
    args = [arg for arg in args if arg != "--share"]
    if len(args) != 1:
        raise SystemExit("usage: marimo_iframe_url.py [--share] NOTEBOOK.py")

    source = Path(args[0]).read_text(encoding="utf-8")
    compressed = LZString().compressToEncodedURIComponent(source)
    if share:
        # A full marimo.app editor tab that students can keep working in.
        print(f"https://marimo.app/#code/{compressed}")
        return
    print(
        "https://marimo.app/"
        "?embed=true&mode=edit&show-chrome=false"
        f"#code/{compressed}"
    )


if __name__ == "__main__":
    main()
