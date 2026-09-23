"""Check L08/L09 examples and regenerate their static teaching figures.

    uv run --locked python scripts/l08_l09_figures.py
    uv run --locked python scripts/l08_l09_figures.py --check-only

No Quarto render or browser is needed. Figures use the notebook calculations.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "activities" / f"{name}.edit.py"
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    return definitions


def save(figure, filename):
    path = ROOT / "assets" / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(path.relative_to(ROOT))


def check(runge, spline, phase, line, arr, lagrange):
    x_data, y_data = lagrange["x_data"], lagrange["y_data"]
    assert len(x_data) == len(y_data) == 7
    for n_points in range(2, len(x_data) + 1):
        selected_x = x_data[:n_points]
        selected_y = y_data[:n_points]
        for i in range(n_points):
            basis_at_nodes = np.array([
                lagrange["lagrange_basis"](x_j, i, selected_x)
                for x_j in selected_x
            ])
            expected = np.zeros(n_points)
            expected[i] = 1.0
            np.testing.assert_allclose(basis_at_nodes, expected, atol=1e-12)
        np.testing.assert_allclose(
            lagrange["lagrange_interpolation"](selected_x, selected_x, selected_y),
            selected_y,
            atol=1e-12,
        )
    assert lagrange["n_points"] == 7
    assert len(lagrange["selected_x"]) == len(lagrange["selected_y"]) == 7

    # Exact at nodes is separate from accuracy between nodes.
    for count in range(5, 26, 2):
        for clustered in (False, True):
            x, y, grid, poly, cubic = runge["runge_curves"](count, clustered)
            np.testing.assert_allclose(poly(x), y, atol=1e-12)
            np.testing.assert_allclose(cubic(x), y, atol=1e-12)
            assert np.all(np.isfinite(poly(grid)))
    assert 1.9 < runge["runge_grid_error"] < 2.0
    x, y, grid, poly, _ = runge["runge_curves"](11, True)
    assert np.max(np.abs(poly(grid)-runge["reference"](grid))) < 0.2
    assert runge["runge_figure"].axes[0].get_ylim()[1] < 1.1
    assert runge["runge_figure"].axes[1].get_ylim()[0] == 1e-6
    gaussian = lambda x: np.exp(-5*x**2)
    for clustered in (False, True):
        x, y, grid, poly, cubic = runge["runge_curves"](21, clustered, gaussian)
        np.testing.assert_allclose(poly(x), y, atol=1e-12)
        np.testing.assert_allclose(cubic(x), y, atol=1e-12)
        assert np.max(np.abs(poly(grid)-gaussian(grid))) < 1e-4

    # The Gaussian case is accurate inside the data range but can be poor when extended.
    x, y, grid, poly, _ = runge["runge_curves"](21, False, gaussian, False)
    assert (grid[0], grid[-1]) == (-1, 1)
    inside_error = np.abs(poly(grid)-gaussian(grid))
    assert np.max(inside_error) < 1e-4
    x, y, grid, poly, _ = runge["runge_curves"](21, False, gaussian, True)
    assert (grid[0], grid[-1]) == (-1.5, 1.5)
    inside = (grid >= -1) & (grid <= 1)
    outside = ~inside
    extrapolated_error = np.abs(poly(grid)-gaussian(grid))
    assert np.max(extrapolated_error[inside]) < 1e-4
    assert np.max(extrapolated_error[outside]) > 10

    np.testing.assert_allclose(spline["cubic"](spline["T"]), spline["conductivity"])
    np.testing.assert_allclose(spline["cubic"](spline["T"][[0, -1]], 2), 0, atol=1e-12)
    assert np.isnan(spline["cubic"](750)) and np.isnan(spline["pchip"](750))
    grid = np.linspace(300, 700, 1001)
    assert np.all(spline["pchip"](grid, 1) >= -1e-12)
    assert np.any(spline["cubic"](grid, 1) < 0)

    np.testing.assert_allclose(phase["saturation_pressure"](280), 55.1290525746, atol=1e-8)
    for T in np.linspace(250, phase["critical_T"]-0.05, 65):
        P = phase["saturation_pressure"](T)
        assert abs(phase["delta_g"](T, P)) < 1e-6
        assert len(phase["stationary_volumes"](T, P)) == 3
    for count in range(4, 11):
        T = np.linspace(250, 295, count)
        P = np.array([phase["saturation_pressure"](t) for t in T])
        for constructor in (CubicSpline, PchipInterpolator):
            interpolant = constructor(T, P, extrapolate=False)
            np.testing.assert_allclose(interpolant(T), P, atol=1e-12)
            assert np.isnan(interpolant(305))
    assert np.isnan(phase["saturation_pressure"](305))
    assert phase["node_residual"] < 1e-10
    assert abs(phase["estimated_P"]-phase["direct_P"]) < 0.01

    strain, stress = line["strain"], line["stress"]
    expected = np.linalg.lstsq(np.column_stack([np.ones_like(strain), strain]), stress, rcond=None)[0]
    np.testing.assert_allclose(line["fitted"].convert().coef, expected, atol=1e-12)
    assert abs(np.sum(line["residuals"])) < 1e-10
    assert abs(np.dot(strain, line["residuals"])) < 1e-10
    assert line["SSE"] > 0
    interpolating_fit = np.polynomial.Polynomial.fit(strain, stress, 6)
    np.testing.assert_allclose(interpolating_fit(strain), stress, atol=1e-8)

    k, T = arr["k"], arr["T"]
    raw_error = lambda predicted: np.sum((k-predicted)**2)
    log_error = lambda predicted: np.sum((np.log(k)-np.log(predicted))**2)
    assert raw_error(arr["k_direct"]) < raw_error(arr["k_linear"])
    assert log_error(arr["k_linear"]) < log_error(arr["k_direct"])
    for guess in ((np.log(1e7), 80), (np.log(1e6), 60), (np.log(1e8), 100)):
        params, _ = curve_fit(arr["arrhenius"], T, k, p0=guess)
        np.testing.assert_allclose(params, arr["parameters"], rtol=2e-5)
    params, _ = curve_fit(arr["arrhenius"], T, k, p0=(arr["log_A"], arr["Q_linear"]), sigma=0.15*k)
    weighted = arr["arrhenius"](T, *params)
    assert np.sum(((k-weighted)/(0.15*k))**2) <= np.sum(((k-arr["k_linear"])/(0.15*k))**2)
    assert abs(params[1]-arr["Q_linear"]) < abs(arr["Q_direct"]-arr["Q_linear"])
    exact_k = arr["arrhenius"](T, np.log(1e7), 80)
    exact_params, _ = curve_fit(arr["arrhenius"], T, exact_k, p0=(np.log(1e6), 70))
    np.testing.assert_allclose(exact_params, [np.log(1e7), 80], rtol=1e-8)
    print("PASS: node residuals, Runge behaviour, spline constraints, phase equilibrium, linear and nonlinear fits.")
    print(f"Arrhenius SSE: linearized {raw_error(arr['k_linear']):.6g}, direct {raw_error(arr['k_direct']):.6g}")
    print(f"Arrhenius log SSE: linearized {log_error(arr['k_linear']):.6g}, direct {log_error(arr['k_direct']):.6g}")


def check_lecture_systems():
    # Three-point Lagrange/Newton example and its coefficient system.
    x = np.array([0., 2., 3.])
    y = np.array([7., 11., 28.])
    coefficients = np.linalg.solve(np.vander(x, 3, increasing=True), y)
    np.testing.assert_allclose(coefficients, [7., -8., 5.])
    first = np.diff(y)/np.diff(x)
    second = (first[1]-first[0])/(x[2]-x[0])
    np.testing.assert_allclose([y[0], first[0], second], [7., 2., 5.])

    # All eight rows shown in the natural-cubic example in L08.
    matrix = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [3, 2, 1, 0, 0, 0, -1, 0],
        [6, 2, 0, 0, 0, -2, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 2, 0, 0],
    ], dtype=float)
    values = np.array([0., 1., 1., 0., 0., 0., 0., 0.])
    coefficients = np.linalg.solve(matrix, values)
    expected = CubicSpline([0., 1., 2.], [0., 1., 0.], bc_type="natural").c.T.ravel()
    np.testing.assert_allclose(coefficients, expected, atol=1e-14)
    np.testing.assert_allclose(matrix @ coefficients, values, atol=1e-14)
    print("PASS: explicit interpolation matrix, divided differences and complete cubic-spline system.")


def divided_difference_figure():
    # Stagger the differences so each arrow pair identifies the two inputs.
    positions = [0.5, 3.2, 6.8, 11.0, 16.0]
    half_widths = [0.2, 0.8, 1.2, 1.6, 2.0]
    fig, ax = plt.subplots(figsize=(14, 4.8), layout="constrained")
    for column, position in enumerate(positions):
        ax.text(position, 5.0, rf"$a_{column}$", ha="center", color="firebrick", fontsize=19)
        for row in range(5-column):
            y_position = 4-row-column/2
            if column == 0:
                label = rf"$y_{row}$"
                ax.text(-0.65, y_position, rf"$x_{row}$", ha="center", va="center", fontsize=16)
            else:
                inputs = ",".join(rf"x_{j}" for j in range(row, row+column+1))
                label = "$f[" + inputs + "]$"
                for source_row in (row, row+1):
                    source_y = 4-source_row-(column-1)/2
                    ax.annotate("", xy=(position-half_widths[column]-0.1, y_position),
                                xytext=(positions[column-1]+half_widths[column-1]+0.1, source_y),
                                arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 1})
            ax.text(position, y_position, label, ha="center", va="center", fontsize=16,
                    color="firebrick" if row == 0 else "black")
    for boundary in (-0.05, 1.65, 4.85, 8.9, 13.5):
        ax.plot([boundary, boundary], [-0.45, 4.6], color="0.7", lw=0.7)
    ax.plot([-1.2, 18.5], [4.6, 4.6], color="0.4", lw=0.8)
    labels = ["Known\nvalues", "First divided\ndifference", "Second divided\ndifference", "Third divided\ndifference", "Fourth divided\ndifference"]
    for position, label in zip(positions, labels):
        ax.text(position, -0.65, label, ha="center", va="top", fontsize=12)
    ax.set(xlim=(-1.3, 18.6), ylim=(-1.5, 5.6))
    ax.axis("off")
    save(fig, "L08-divided-differences.png")


def extra_figures(phase, line):
    # Opening phase diagram: only a sparse table and the question between nodes.
    T, P = phase["sample_T"], phase["sample_P"]
    grid = np.linspace(T[0], T[-1], 300)
    guide = PchipInterpolator(T, P)(grid)
    x_min, x_max = float(T[0]), float(T[-1])
    y_min, y_max = 20, 84
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.fill_between(grid, y_min, guide, color="tab:blue", alpha=0.08)
    ax.fill_between(grid, guide, y_max, color="tab:orange", alpha=0.10)
    ax.scatter(T, P, c="black", zorder=4, label="Calculated coexistence states")
    ax.text(260, 62, "Liquid", color="saddlebrown", fontsize=14)
    ax.text(280, 29, "Gas", color="tab:blue", fontsize=14)
    ax.annotate("Where is the boundary\nbetween these points?", xy=(280, 55.1), xytext=(267, 75), arrowprops={"arrowstyle": "->"}, ha="center", fontsize=11)
    ax.set(xlabel="Temperature (K)", ylabel="Pressure (bar)", xlim=(x_min, x_max), ylim=(y_min, y_max), title="The L07 vdW CO₂ model")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15)
    save(fig, "L08-phase-points.png")

    x = np.arange(5.)
    y = np.array([0.8, 1.5, 1.2, 2.0, 1.7])
    cubic = CubicSpline(x, y, bc_type="natural")
    fig, ax = plt.subplots(figsize=(9, 3.3), layout="constrained")
    for i in range(4):
        segment = np.linspace(x[i], x[i+1], 100)
        ax.plot(segment, cubic(segment), lw=5, color=plt.cm.tab10(i), solid_capstyle="round")
    ax.scatter(x, y, s=100, facecolor="white", edgecolor="black", zorder=5)
    ax.scatter(x, y, s=12, color="black", zorder=6)
    for i in range(5):
        ax.text(x[i], y[i]-0.24, f"$x_{i}$", ha="center", fontsize=12)
    ax.annotate("Pin = supplied data point", xy=(1, 1.5), xytext=(0.5, 2.35), arrowprops={"arrowstyle": "->"}, ha="center")
    ax.annotate("Flexible strip: cubic pieces", xy=(2.6, cubic(2.6)), xytext=(3.1, 2.6), arrowprops={"arrowstyle": "->"}, ha="center")
    ax.text(0, 0.17, "Free end rotation\n$S''(x_0)=0$", ha="center")
    ax.text(4, 0.75, "Free end rotation\n$S''(x_4)=0$", ha="center")
    ax.set(xlim=(-0.55, 4.5), ylim=(-0.1, 2.9), xlabel="x", ylabel="y")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "L08-spline-pins.png")

    # Subtract a fixed elastic trend so the small scatter remains visible.
    x, y = line["strain"], line["stress"]
    grid = np.linspace(0, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].plot(grid, CubicSpline(x, y)(grid)-70*grid, label="Cubic interpolant")
    axes[1].plot(grid, line["fitted"](grid)-70*grid, label="Least-squares line")
    axes[1].vlines(x, line["fitted"](x)-70*x, y-70*x, color="tab:red", label="Residuals")
    for ax, title in zip(axes, ("Pass through every point", "Represent the overall trend")):
        ax.scatter(x, y-70*x, color="black", s=25, zorder=4, label="Same measurements")
        ax.set(xlabel="Strain, x (mm/m)", ylabel="Stress − 70x (MPa)", ylim=(-5, 6), title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Subtract the same 70x trend to see the scatter")
    save(fig, "L09-interpolation-vs-fitting.png")

    lsr_landscape_figure()


def lsr_landscape_figure():
    # Illustrative positive-definite quadratic, independent of lecture data.
    # The cross term tilts the elliptical contours in the two parameters.
    A0, A1 = np.meshgrid(np.linspace(-3, 4, 301), np.linspace(-4, 3, 301))
    u, v = A0 - 0.5, A1 + 0.5
    error = 2 + 3*u**2 + 3.6*u*v + 2*v**2
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    contours = ax.contour(A0, A1, error, levels=[3, 5, 9, 17, 29, 45], cmap="viridis")
    ax.clabel(contours, fontsize=9, fmt="%g")
    ax.scatter(0.5, -0.5, marker="*", s=180, color="tab:red", zorder=3, label="Minimum")
    ax.set(xlabel=r"$a_0$", ylabel=r"$a_1$", aspect="equal",
           title="Typical landscape of residual\n(summed square error, SSE) in LSR")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "L09-error-landscape.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    names = ["l08_runge", "l08_scipy_interpolation", "l08_co2_interpolation", "l09_least_squares", "l09_arrhenius", "l08_lagrange"]
    definitions = [load(name) for name in names]
    check(*definitions)
    check_lecture_systems()
    if not args.check_only:
        for data, key, filename in zip(definitions,
                ["runge_figure", "spline_figure", "phase_figure", "regression_figure", "arrhenius_figure", "lagrange_figure"],
                ["L08-runge.png", "L08-scipy-interpolation.png", "L08-co2-interpolation.png", "L09-least-squares.png", "L09-arrhenius.png", "L08-lagrange-terms.png"]):
            save(data[key], filename)
        extra_figures(definitions[2], definitions[3])
        divided_difference_figure()
    plt.close("all")


if __name__ == "__main__":
    main()
