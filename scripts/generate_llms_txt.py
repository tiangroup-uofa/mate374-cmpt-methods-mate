#!/usr/bin/env python3
"""Generate an agent-facing llms.txt from the rendered Quarto search index."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urljoin


SITE_URL = "https://tiangroup-uofa.github.io/mate374-cmpt-methods-mate/"
EXCLUDED_PREFIXES = (
    "units/archive/",
)
EXCLUDED_PATHS = {
    "units/01/media-backlog.html",
}


def output_directory() -> Path:
    configured = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
    return Path(configured) if configured else Path("_site")


def canonical_url(href: str) -> str:
    if href == "index.html":
        href = ""
    elif href.endswith("/index.html"):
        href = href[: -len("index.html")]
    return urljoin(SITE_URL, href)


def category(href: str) -> str:
    if href.startswith("units/") and "/L" in href:
        return "Lecture notes"
    if href.startswith("units/"):
        return "Unit and reference pages"
    if href.startswith(("assignments/", "project/", "seminars/")):
        return "Activities and assessment"
    return "Course information"


def main() -> None:
    output_dir = output_directory()
    search_path = output_dir / "search.json"
    if not search_path.is_file():
        print(f"Agent index skipped: {search_path} does not exist.")
        return

    records = json.loads(search_path.read_text(encoding="utf-8"))
    pages: dict[str, str] = {}

    for record in records:
        href = str(record.get("href", "")).split("#", 1)[0]
        title = str(record.get("title", "")).strip()
        if not href.endswith(".html") or not title:
            continue
        if href in EXCLUDED_PATHS or href.startswith(EXCLUDED_PREFIXES):
            continue
        pages.setdefault(href, title)

    grouped: dict[str, list[tuple[str, str]]] = {
        "Course information": [],
        "Lecture notes": [],
        "Unit and reference pages": [],
        "Activities and assessment": [],
    }
    for href, title in pages.items():
        grouped[category(href)].append((href, title))

    lines = [
        "# MATE 374 · Computational Methods in Materials Engineering",
        "",
        "> Agent-facing index of the current public course website. This file is generated from the rendered site; archived and internal planning pages are excluded.",
        "",
        "## Source policy",
        "",
        "- Use the current pages listed here before historical course materials.",
        "- Canvas remains authoritative for announcements and submissions.",
        "- The current syllabus remains authoritative for policies and assessment rules.",
        "- A published page does not necessarily mean that its lecture has already been taught; ask the student when that distinction matters.",
        "- Do not infer or invent URLs that are not listed here.",
    ]

    for heading, entries in grouped.items():
        if not entries:
            continue
        lines.extend(["", f"## {heading}", ""])
        for href, title in sorted(entries, key=lambda item: item[0]):
            lines.append(f"- [{title}]({canonical_url(href)})")

    lines.extend(
        [
            "",
            "## Official software documentation",
            "",
            "- [Python 3.12](https://docs.python.org/3.12/)",
            "- [NumPy 2.0](https://numpy.org/doc/2.0/)",
            "- [SciPy](https://docs.scipy.org/doc/scipy/)",
            "- [marimo](https://docs.marimo.io/)",
            "",
        ]
    )

    destination = output_dir / "llms.txt"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Agent index created: {destination} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
