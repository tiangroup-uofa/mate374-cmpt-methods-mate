"""Check the S06 calculations and regenerate their static notebook figures.

    uv run --locked python scripts/s06_figures.py [--check-only]
"""
import argparse
import os
from pathlib import Path
import runpy

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check_hand_work():
    A1 = np.array([[2, 1, -1], [4, 5, 0], [-2, 2, 7]], dtype=float)
    b1 = np.array([5, 14, -5], dtype=float)
    aug = np.column_stack([A1, b1])
    aug[1] -= 2*aug[0]
    aug[2] += aug[0]
    aug[2] -= aug[1]
    np.testing.assert_allclose(aug, [[2, 1, -1, 5], [0, 3, 2, 4], [0, 0, 4, -4]])
    np.testing.assert_allclose(A1 @ np.array([1, 2, -1]), b1)

    A2 = np.array([[0, 2, 1], [2, 1, -1], [4, 4, 1]], dtype=float)
    b2 = np.array([0, -1, 2], dtype=float)
    aug = np.column_stack([A2, b2])
    aug[[0, 2]] = aug[[2, 0]]
    aug[1] -= 0.5*aug[0]
    aug[[1, 2]] = aug[[2, 1]]
    aug[2] += 0.5*aug[1]
    np.testing.assert_allclose(aug, [[4, 4, 1, 2], [0, 2, 1, 0], [0, 0, -1, -2]])
    np.testing.assert_allclose(A2 @ np.array([1, -1, 2]), b2)


def check_diffusion(d):
    assert d["side"] == 100
    assert d["N"] == 10_000
    assert np.max(d["scaled_residuals"]) < 1e-12
    assert np.min(d["concentrations"]) > 0
    assert d["concentrations"].shape == (100, 100, 2)
    assert d["dense_repeated_ops"] > d["dense_lu_ops"] > d["dense_cholesky_ops"]
    assert d["dense_cholesky_ops"] > 1e6*d["fast_poisson_work"]
    np.testing.assert_allclose(d["L_small"] @ d["L_small"].T, d["A_small"], atol=1e-14)
    # Independently check the displayed 2D matrix against a dense solve.
    rhs = np.ones(d["A_small"].shape[0])
    solution = np.linalg.solve(d["A_small"], rhs)
    np.testing.assert_allclose(d["A_small"] @ solution, rhs, atol=1e-12)
    print("Scaled residuals:", d["scaled_residuals"])
    print(f"Optimized 2D solve time: {d['solve_seconds']:.3g} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_hand_work()
    app = runpy.run_path(str(ROOT / "activities/s06_diffusion_cholesky.edit.py"))["app"]
    _, definitions = app.run()
    d = dict(definitions)
    check_diffusion(d)
    if not args.check_only:
        for key, filename in [("pattern_figure", "S06-cholesky-pattern.png"),
                              ("profile_figure", "S06-diffusion-profiles.png")]:
            d[key].savefig(ROOT / "assets" / filename, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved assets/{filename}")
    plt.close("all")
    print("S06 checks passed.")


if __name__ == "__main__":
    main()
