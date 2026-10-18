"""Fail the website build if local-only examination files enter the output."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    site = ROOT / "_site"
    forbidden = []
    for path in site.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(site)
        if "instructor-private" in rel.parts or "midterm-exam" in path.name:
            forbidden.append(str(rel))
        # Real-exam starters and all blank exam notebooks are deliberately
        # unlisted: no links on course pages, search, or the agent index.
        if rel.parts[0] != "wasm-local" and path.suffix in {".html", ".json", ".txt"}:
            content = path.read_text(errors="replace")
            if any(name in content for name in (
                "midterm_real_q2.html", "midterm_real_q3.html",
                "midterm_real_blank_q2.html", "midterm_real_blank_q3.html",
                "midterm_mock_blank_q2.html", "midterm_mock_blank_q3.html",
            )):
                forbidden.append(f"{rel}: links to unlisted exam notebooks")
    if forbidden:
        sys.exit("Exam publication check failed:\n" + "\n".join(forbidden))
    print("Exam publication check passed: no private exam files or unlisted-notebook links.")


if __name__ == "__main__":
    main()
