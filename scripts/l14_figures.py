"""Check the L14 minimizer notebooks and draw the static L14 figures.

    uv run --locked python scripts/l14_figures.py [--check-only]
"""
import os

# Tiny optimization problems run far slower with threaded OpenBLAS; the browser is single-threaded.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DPI = 300


def load(name):
    app = runpy.run_path(str(ROOT / "activities" / f"{name}.edit.py"))["app"]
    _, definitions = app.run()
    return dict(definitions)


def check_paths(d):
    # Default start (0.3, 0.1), l0 = 1.2a: Newton converges to the saddle, the others to a minimum.
    paths = d["paths"]
    assert np.allclose(paths["Newton"][-1], [0, 0], atol=1e-8)
    for name in ("Gradient descent", "Nelder-Mead", "CG", "BFGS", "L-BFGS-B"):
        assert np.allclose(paths[name][-1], [0, np.sqrt(1.2**2 - 1)], atol=1e-3), name
    # From (1.5, 0.05), Newton finds the other stationary point on the axis, (l0, 0).
    newton = d["newton"](np.array([1.5, 0.05]), 1.2)
    assert np.allclose(newton[-1], [1.2, 0], atol=1e-8)
    print("Steps from (0.3, 0.1):", {k: len(v) - 1 for k, v in paths.items()})


def check_ase(d):
    E = {k: h[-1, 0] for k, h in d["histories"].items()}
    # 13 argon atoms relax to the icosahedron, -44.3268 eps = -0.456567 eV (rc = 50 Å shift is tiny).
    assert all(abs(e + 0.456567) < 1e-4 for e in E.values()), E
    print("Ar13 steps:", {k: len(h) - 1 for k, h in d["histories"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    paths = load("l14_minimizer_paths")
    ase_relax = load("l14_ase_relax")
    check_paths(paths)
    check_ase(ase_relax)
    if not args.check_only:
        for d, key, name in [(paths, "paths_figure", "L14-minimizer-paths.png"),
                             (ase_relax, "history_figure", "L14-ase-relaxation.png")]:
            d[key].savefig(ASSETS / name, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved assets/{name}")
    plt.close("all")
    print("L14 checks passed.")


if __name__ == "__main__":
    main()
