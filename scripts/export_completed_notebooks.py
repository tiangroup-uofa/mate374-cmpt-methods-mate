"""Export unlisted completed activity notebooks, linked only from answer PDFs.

These are PUBLIC resources, not access-controlled answers. They are omitted
from student navigation, Quarto search, and llms.txt. The opaque path and
noindex metadata reduce accidental discovery; they do not provide secrecy.
"""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _marimo_export as exporter

OUTPUT = ROOT / "completed-notebooks" / "8f7c2e91"


def main():
    notebooks = sorted((ROOT / "activities").glob("*_completed.edit.py"))
    if not notebooks:
        raise RuntimeError("No completed notebooks found")
    fingerprint = hashlib.sha256(
        (exporter.export_fingerprint(ROOT, notebooks) + Path(__file__).read_text()).encode()
    ).hexdigest()
    if exporter.cache_is_current(OUTPUT, notebooks, fingerprint):
        print("Completed notebook exports are current.")
        return
    exporter.export_all(ROOT, notebooks, OUTPUT, fingerprint)
    for page in OUTPUT.glob("*.html"):
        html = page.read_text()
        if "</head>" not in html:
            raise RuntimeError(f"Missing HTML head: {page}")
        page.write_text(html.replace(
            "</head>", '<meta name="robots" content="noindex, nofollow, noarchive">\n</head>', 1
        ))
    print(f"Exported {len(notebooks)} completed notebooks to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
