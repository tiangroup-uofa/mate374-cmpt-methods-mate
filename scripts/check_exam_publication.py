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
        # Real-exam starter notebooks must not be linked from public course pages.
        if rel.parts[0] != "wasm-local" and path.suffix in {".html", ".json", ".txt"}:
            content = path.read_text(errors="replace")
            if any(name in content for name in (
                "midterm_real_q2.html", "midterm_real_q3.html",
            )):
                forbidden.append(f"{rel}: links to unlisted real-exam notebooks")
    if forbidden:
        sys.exit("Exam publication check failed:\n" + "\n".join(forbidden))
    print("Exam publication check passed: no private exam files or unlisted real-exam notebook links.")


if __name__ == "__main__":
    main()
