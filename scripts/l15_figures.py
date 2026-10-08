"""Check the L15 minimizer and ASE notebooks and draw the static L15 figures.

    uv run --locked python scripts/l15_figures.py [--check-only]
"""
import os

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


@matplotlib.rc_context({"font.size": 14, "font.family": "Arial"})
def energy_sections(d):
    energy = d["energy"]
    points = np.linspace(-0.85, 0.85, 181)
    X, Y = np.meshgrid(points, points)
    Z = np.array([[energy(np.array([x, y]), 1.2) for x in points] for y in points])
    fig, (ax, cut) = plt.subplots(1, 2, figsize=(9, 4.3), layout="constrained")
    ax.contour(X, Y, Z, levels=[0.002, 0.01, 0.025, 0.04, 0.08, 0.16, 0.32], colors="#8a8f98")
    ax.axhline(0, color="#231f20", ls="--")
    ax.axvline(0, color="#b5473a", ls="--")
    ax.plot([0, 0], [-np.sqrt(0.44), np.sqrt(0.44)], "o", color="#b5473a")
    ax.plot(0, 0, "x", color="#231f20", ms=9)
    ax.set(xlabel="x / a", ylabel="y / a", aspect="equal", title="Energy contours")
    cut.plot(points, [energy(np.array([s, 0]), 1.2) for s in points],
             color="#231f20", label="Along x (y = 0)")
    cut.plot(points, [energy(np.array([0, s]), 1.2) for s in points],
             color="#b5473a", label="Along y (x = 0)")
    cut.plot(0, 0.04, "x", color="#231f20", ms=9)
    cut.set(xlabel="Displacement from origin / a", ylabel="Energy / (k a²)", ylim=(0, 0.22))
    cut.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    paths = load("l15_minimizer_paths")
    ase_relax = load("l15_ase_relax")
    check_paths(paths)
    check_ase(ase_relax)
    if not args.check_only:
        figures = [(energy_sections(paths), "L15-energy-sections.png"),
                   (paths["paths_figure"], "L15-minimizer-paths.png"),
                   (ase_relax["history_figure"], "L15-ase-relaxation.png")]
        for fig, name in figures:
            fig.savefig(ASSETS / name, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved assets/{name}")
    plt.close("all")
    print("L15 checks passed.")


if __name__ == "__main__":
    main()
