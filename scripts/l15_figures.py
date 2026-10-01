"""Check the L15 Hessian notebooks and draw the static L15 figures.

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


def check(spring, cluster):
    # theta = 45°, l0 = 1.2a: H = k [[0.8, 1.2], [1.2, 0.8]], eigenvalues -0.4k and 2k.
    np.testing.assert_allclose(spring["H0"], [[0.8, 1.2], [1.2, 0.8]], atol=1e-6)
    np.testing.assert_allclose(spring["values"], [-0.4, 2.0], atol=1e-6)
    assert not spring["is_positive_definite"](spring["H0"])
    assert spring["is_positive_definite"](spring["H_min"])
    # A1 square: two negative modes; following mode 1 passes the rhombus saddle to the tetrahedron.
    assert cluster["square_counts"] == {"negative": 2, "zero": 6, "positive": 4}
    energies = [cluster["lj_energy"](X) for X in cluster["follow_path"]]
    np.testing.assert_allclose(energies, [-4.480620, -5.073421, -6.0], atol=1e-6)
    print("Square eigenvalues:", np.round(cluster["square_values"], 3))


def spring_figure(spring):
    energy, fixed_atoms, hessian = spring["energy"], spring["fixed_atoms"], spring["hessian"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    xs = np.linspace(-1.5, 1.5, 241)
    Xg, Yg = np.meshgrid(xs, xs)
    for ax, theta in zip(axes, (0, 45)):
        fixed = fixed_atoms(theta)
        Z = np.array([[energy(np.array([x, y]), fixed, 1.2) for x in xs] for y in xs])
        ax.contourf(Xg, Yg, np.log10(Z + 1e-3), levels=30, cmap="Greys_r", alpha=0.6)
        ax.contour(Xg, Yg, Z, levels=np.linspace(0, 1.0, 11), colors="white", linewidths=0.5)
        ax.plot(*fixed.T, "s", color="black", ms=9)
        H = hessian(np.zeros(2), fixed, 1.2)
        values, vectors = np.linalg.eigh(H)
        for lam, v, colour in zip(values, vectors.T, ["tab:red", "tab:blue"]):
            ax.annotate("", xy=0.6*v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colour, lw=2.5))
            ax.plot([], [], color=colour, lw=2.5, label=f"λ = {lam:+.2f}k")
        title = (f"θ = {theta}°:  H = k[[{H[0, 0]:.1f}, {H[0, 1]:.1f}], [{H[1, 0]:.1f}, {H[1, 1]:.1f}]]")
        ax.set(xlabel="x / a", ylabel="y / a", aspect="equal", title=title)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    spring, cluster = load("l15_saddle_eigen"), load("l15_cluster_check")
    check(spring, cluster)
    if not args.check_only:
        figures = [(spring_figure(spring), "L15-compressed-spring.png"),
                   (cluster["square_figure"], "L15-square-eigenvalues.png")]
        for fig, name in figures:
            fig.savefig(ASSETS / name, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved assets/{name}")
    plt.close("all")
    print("L15 checks passed.")


if __name__ == "__main__":
    main()
