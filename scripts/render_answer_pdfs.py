"""Render answer PDFs separately and stage only PDFs for the course website.

The answer source project has no website navigation or HTML output. Staged PDFs
are public resources, not access-controlled files, even when no page links them.
"""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run(["quarto", "render", "answer-keys", "--to", "pdf"], cwd=ROOT, check=True)
    source = ROOT / "answer-keys/render/A1/A1-answers.pdf"
    target = ROOT / "assignments/A1/A1-answers.pdf"
    if not source.is_file():
        raise FileNotFoundError(f"Answer PDF was not produced: {source}")
    shutil.copy2(source, target)
    print(f"Staged {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
