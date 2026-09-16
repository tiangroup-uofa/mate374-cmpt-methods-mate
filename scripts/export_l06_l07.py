"""Export only the three L06/L07 demos, without executing notebooks at build time.

Run: uv run python scripts/export_l06_l07.py
Live/native checks are separate; this never runs the whole-course exporter.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _marimo_export import enable_browser_auto_run, expose_browser_save

NOTEBOOKS = [
    "l06_open_methods.edit.py",
    "l06_fixed_point.edit.py",
    "l07_co2_free_energy.edit.py",
]


def main():
    output = ROOT / "wasm-local"
    output.mkdir(exist_ok=True)
    for name in NOTEBOOKS:
        source = ROOT / "activities" / name
        target = output / name.replace(".edit.py", ".html")
        command = [
            sys.executable, "-m", "marimo", "export", "html-wasm",
            str(source), "--mode", "edit", "--no-execute", "--force",
            "-o", str(target),
        ]
        print(f"Exporting {name} (20-second limit)", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True, timeout=20)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"Stopped slow export: {name}. Continue in molab.")
        enable_browser_auto_run(target, source)
        expose_browser_save(target)
        shutil.copy2(source, output / name)
        print(target, flush=True)


if __name__ == "__main__":
    main()
