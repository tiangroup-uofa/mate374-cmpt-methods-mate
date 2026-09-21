"""Regenerate scripted course figures from source and saved data (no benchmarks).

Run: uv run --locked python scripts/render_figures.py
Also run by the full Quarto profile before rendering pages and answer PDFs.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = (
    "binary_place_values.py",
    "polygon_precision.py",
    "l03_error_figures.py",
    "benchmark_scaling.py",
    "wasm_scaling.py",
    "l04_final_scaling.py",
    "l05_root_figures.py",
    "l06_l07_figures.py",
    "l08_l09_figures.py",
)


def main():
    env = {**os.environ, "MPLBACKEND": "Agg"}
    for script in FIGURE_SCRIPTS:
        print(f"Generating figures: {script}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=ROOT, env=env, check=True,
        )


if __name__ == "__main__":
    main()
