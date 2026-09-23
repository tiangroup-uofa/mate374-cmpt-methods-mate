"""Temporary focused WASM export while L08/L09 await push approval.

Reuse the course exporter, cache only these notebooks, and leave other notebooks
in wasm-local untouched. The full profile still invokes the complete exporter.
"""
from pathlib import Path
import fcntl
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _marimo_export import (  # noqa: E402
    MANIFEST_NAME,
    cache_is_current,
    export_all,
    export_fingerprint,
)


def export_focused():
    notebooks = sorted((ROOT / "activities").glob("l0[89]_*.edit.py"))
    if len(notebooks) != 8:
        raise SystemExit("Expected the eight L08/L09 notebooks; review the focused export list.")
    cache = ROOT / ".quarto" / "l08-l09-wasm"
    fingerprint = export_fingerprint(ROOT, notebooks)
    if not cache_is_current(cache, notebooks, fingerprint):
        export_all(ROOT, notebooks, cache, fingerprint)
    else:
        print("L08/L09 WASM exports are current; skipping export.")
    target = ROOT / "wasm-local"
    target.mkdir(exist_ok=True)
    for source in cache.iterdir():
        if source.name == MANIFEST_NAME:
            continue
        if source.is_dir():
            shutil.copytree(source, target / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target / source.name)


def main():
    # Preview can invoke hooks more than once; share the full exporter's lock.
    lock_path = ROOT / ".quarto" / "marimo-export.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        export_focused()


if __name__ == "__main__":
    main()
