"""Validate S04 worksheets and regenerate their static figures at 300 dpi.

    uv run --locked python scripts/s04_figures.py
    uv run --locked python scripts/s04_figures.py --check-only
"""
import argparse
from pathlib import Path
import re
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    app = runpy.run_path(str(ROOT / "activities" / f"s04_{name}.edit.py"))["app"]
    _, definitions = app.run()
    return definitions


def check_spline(d):
    x, y, pieces = d["x"], d["y"], d["pieces"]
    assert d["A"].shape == (12, 12)
    assert np.linalg.matrix_rank(d["A"]) == 12
    np.testing.assert_allclose(d["A"] @ d["coefficients"], d["rhs"], atol=1e-11)
    np.testing.assert_allclose(d["endpoint_errors"], 0, atol=1e-11)
    np.testing.assert_allclose(d["slope_jumps"], 0, atol=1e-11)
    np.testing.assert_allclose(2 * pieces[0, 0], 0, atol=1e-12)
    expected = [[0, 4/3, -17/3], [-13/48, 175/24, -615/16],
                [1/18, -5/2, 35], [1/16, -11/4, 149/4]]
    np.testing.assert_allclose(pieces, expected, atol=1e-11)
    np.testing.assert_allclose(d["estimate"], 10.483958333333333, atol=1e-11)
    np.testing.assert_allclose(np.polyval(pieces[3], 20), 7.25, atol=1e-11)
    knots = np.r_[np.repeat(x[0], 3), x[1:-1], np.repeat(x[-1], 3)]
    quadratic_reference = make_interp_spline(
        x, y, k=2, t=knots, bc_type=([(2, 0.0)], None)
    )
    for i in range(4):
        grid = np.linspace(x[i], x[i + 1], 101)
        np.testing.assert_allclose(np.polyval(pieces[i], grid),
                                   quadratic_reference(grid), atol=1e-11)
    np.testing.assert_allclose(d["cubic"](12.7), 10.118896381578947)
    assert np.isnan(d["cubic"](23))

    # Independent check with zero curvature at the right end instead.
    right_A = d["A"].copy()
    right_A[-1] = 0
    right_A[-1, 9] = 2
    right_pieces = np.linalg.solve(right_A, d["rhs"]).reshape(4, 3)
    np.testing.assert_allclose(right_pieces[3, 0], 0, atol=1e-12)
    assert abs(right_pieces[0, 0]) > 0.01
    right_reference = make_interp_spline(
        x, y, k=2, t=knots, bc_type=(None, [(2, 0.0)])
    )
    for i in range(4):
        grid = np.linspace(x[i], x[i + 1], 101)
        np.testing.assert_allclose(np.polyval(right_pieces[i], grid),
                                   right_reference(grid), atol=1e-11)


def check_fit(d):
    normal_matrix = np.array([[d["N"], d["Sx"]], [d["Sx"], d["Sxx"]]])
    normal_rhs = np.array([d["Sy"], d["Sxy"]])
    np.testing.assert_allclose(normal_matrix @ d["log_parameters"], normal_rhs, atol=1e-10)
    np.testing.assert_allclose(np.linalg.solve(normal_matrix, normal_rhs),
                               d["log_parameters"], atol=1e-10)
    np.testing.assert_allclose(d["log_parameters"],
                               np.polyfit(d["X"], d["Y"], 1)[::-1], atol=1e-10)
    np.testing.assert_allclose(d["log_parameters"], [19.86287982351404, -4.42437508701609])
    np.testing.assert_allclose(d["direct_parameters"], [17.93608343, -3.81552213], rtol=1e-6)
    np.testing.assert_allclose(d["sse_log_fit"], 33424.00691556, rtol=1e-7)
    np.testing.assert_allclose(d["sse_direct_fit"], 19.39274710, rtol=1e-7)
    np.testing.assert_allclose(d["log_sse_log_fit"], 0.31926063698, rtol=1e-7)
    np.testing.assert_allclose(d["log_sse_direct_fit"], 1.109498656, rtol=1e-6)
    assert d["sse_direct_fit"] < d["sse_log_fit"]
    assert d["log_sse_log_fit"] < d["log_sse_direct_fit"]
    assert d["residual_direct"][-1] / d["mu"][-1] < -1.8
    model = d["viscosity"]
    np.testing.assert_allclose(model(45, *d["log_parameters"]), 20.50722246)
    np.testing.assert_allclose(model(45, *d["direct_parameters"]), 30.31608554)

    # Independently check a second initial guess and relative-residual weighting.
    alternate, _ = curve_fit(model, d["T"], d["mu"], p0=[18., -4.], maxfev=10000)
    np.testing.assert_allclose(alternate, d["direct_parameters"], rtol=1e-5)
    weighted, _ = curve_fit(model, d["T"], d["mu"], p0=d["log_parameters"],
                            sigma=d["mu"], maxfev=10000)
    relative = (d["mu"] - model(d["T"], *weighted)) / d["mu"]
    assert np.sum(relative**2) < np.sum((d["residual_log"] / d["mu"])**2)
    assert not np.allclose(weighted, d["log_parameters"])
    print("Relative-residual fit [alpha, m]:", weighted)


def check_copyable_code(spline, fitting):
    page = (ROOT / "seminars/S04-interpolation-fitting/index.qmd").read_text()
    blocks = re.findall(r"```\{\.python[^\n]*\}\n(.*?)\n```", page, flags=re.S)
    # The page explicitly supplies these through the embedded notebooks.
    scope = dict(np=np, CubicSpline=CubicSpline, make_interp_spline=make_interp_spline,
                 curve_fit=curve_fit, x=spline["x"], y=spline["y"],
                 T=fitting["T"], mu=fitting["mu"])
    for index, block in enumerate(blocks, start=1):
        exec(compile(block, f"S04-code-block-{index}", "exec"), scope)
    np.testing.assert_allclose(scope["pieces"], spline["pieces"])
    np.testing.assert_allclose(scope["log_parameters"], fitting["log_parameters"])
    np.testing.assert_allclose(scope["direct_parameters"], fitting["direct_parameters"])
    print(f"All {len(blocks)} copyable Python blocks executed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    spline = load("quadratic_spline")
    fitting = load("viscosity_fitting")
    check_spline(spline)
    check_fit(fitting)
    check_copyable_code(spline, fitting)
    if not args.check_only:
        for fig, name in [(spline["spline_figure"], "S04-quadratic-spline.png"),
                          (fitting["fitting_figure"], "S04-viscosity-fitting.png")]:
            fig.savefig(ROOT / "assets" / name, dpi=300, bbox_inches="tight")
            print(f"Saved assets/{name}")
    plt.close("all")
    print("S04 numerical checks passed.")


if __name__ == "__main__":
    main()
