# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import BarycentricInterpolator, CubicSpline
    return BarycentricInterpolator, CubicSpline, mo, np, plt


@app.cell(hide_code=True)
def controls(mo):
    case_choice = mo.ui.dropdown(
        [
            "Runge function: interpolation inside the data range",
            "Gaussian: extrapolation outside the data range",
        ],
        value="Runge function: interpolation inside the data range",
        label="Case",
    )
    point_count = mo.ui.slider(5, 25, step=2, value=11, label="Number of data points", show_value=True)
    spacing = mo.ui.dropdown(["Equally spaced", "Clustered near the ends"], value="Equally spaced", label="Point locations")
    show_extrapolated = mo.ui.checkbox(value=False, label="Show extrapolated regions")
    mo.vstack([case_choice, mo.hstack([point_count, spacing, show_extrapolated], wrap=True)])
    return case_choice, point_count, show_extrapolated, spacing


@app.cell
def model(case_choice, np):
    if case_choice.value.startswith("Runge"):
        def reference(x):
            return 1 / (1 + 25 * x**2)
        case_name = "Runge function"
        reference_formula = r"\frac{1}{1+25x^2}"
    else:
        def reference(x):
            return np.exp(-5 * x**2)
        case_name = "Gaussian function"
        reference_formula = r"e^{-5x^2}"
    return case_name, reference, reference_formula


@app.cell(hide_code=True)
def experiment(BarycentricInterpolator, CubicSpline, np, reference):
    def runge_curves(count, clustered=False, function=None, show_extrapolated=False):
        function = reference if function is None else function
        x = np.sort(np.cos(np.linspace(0, np.pi, count))) if clustered else np.linspace(-1, 1, count)
        y = function(x)
        polynomial = BarycentricInterpolator(x, y, random_state=374)
        spline = CubicSpline(x, y, bc_type="natural", extrapolate=False)
        plot_limit = 1.5 if show_extrapolated else 1.0
        grid = np.linspace(-plot_limit, plot_limit, 2001)
        return x, y, grid, polynomial, spline
    return (runge_curves,)


@app.cell(hide_code=True)
def plot(case_name, mo, np, plt, point_count, reference, reference_formula, runge_curves, show_extrapolated, spacing):
    x_nodes, y_nodes, x_grid, global_poly, local_spline = runge_curves(
        point_count.value,
        spacing.value == "Clustered near the ends",
        show_extrapolated=show_extrapolated.value,
    )
    runge_figure, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    _axes[0].plot(x_grid, reference(x_grid), "k--", label="Known function")
    _axes[0].plot(x_grid, global_poly(x_grid), label=f"Degree {len(x_nodes)-1} polynomial")
    _axes[0].plot(x_grid, local_spline(x_grid), label="Natural cubic spline")
    _axes[0].scatter(x_nodes, y_nodes, color="black", s=20, zorder=4, label="Data in [-1, 1]")
    if show_extrapolated.value:
        _axes[0].axvspan(-1.5, -1, color="0.5", alpha=0.12, label="Extrapolated region")
        _axes[0].axvspan(1, 1.5, color="0.5", alpha=0.12)
        _axes[0].axvline(-1, color="0.35", ls=":", lw=0.9)
        _axes[0].axvline(1, color="0.35", ls=":", lw=0.9)

    _data_grid = np.linspace(-1, 1, 1001)
    _data_values = reference(_data_grid)
    _data_min = float(np.min(_data_values))
    _data_max = float(np.max(_data_values))
    _data_margin = 0.05 * max(_data_max - _data_min, 1e-12)
    _axes[0].set(
        xlabel="x",
        ylabel="y",
        title=f"{case_name}: left scale follows the data range",
        ylim=(_data_min - _data_margin, _data_max + _data_margin),
    )

    _largest_error = 0.0
    _error_curves = []
    for _curve, _label in [(global_poly, "Global polynomial"), (local_spline, "Cubic spline")]:
        _error = np.abs(_curve(x_grid)-reference(x_grid))
        _finite = np.isfinite(_error)
        if np.any(_finite):
            _largest_error = max(_largest_error, float(np.max(_error[_finite])))
        _error_curves.append((_error, _label))
    for _error, _label in _error_curves:
        _axes[1].semilogy(
            x_grid,
            np.where(np.isfinite(_error) & (_error >= 1e-6), _error, np.nan),
            label=_label,
        )
    _error_title = "Error inside and outside the data range" if show_extrapolated.value else "Error inside the data range"
    _axes[1].set(
        xlabel="x",
        ylabel="Absolute error",
        title=f"{_error_title} (shown ≥ 10⁻⁶)",
        ylim=(1e-6, max(1e-5, 1.2*_largest_error)),
    )
    _range_label = " and extrapolated regions" if show_extrapolated.value else ""
    runge_figure.suptitle(rf"{case_name}: $f(x)={reference_formula}$ on $[-1,1]${_range_label}")
    for _ax in _axes:
        _ax.grid(alpha=0.2)
        _ax.legend(fontsize=8)

    _global_error = np.abs(global_poly(x_grid)-reference(x_grid))
    _inside = (x_grid >= -1) & (x_grid <= 1)
    runge_node_error = float(np.max(np.abs(global_poly(x_nodes)-y_nodes)))
    runge_grid_error = float(np.max(_global_error[_inside]))
    runge_extrapolation_error = float(np.max(_global_error[~_inside])) if np.any(~_inside) else np.nan
    if case_name == "Runge function":
        _message = f"**Runge case:** the polynomial matches the supplied points to **{runge_node_error:.1e}**, but its largest error inside the data range is **{runge_grid_error:.3g}**."
    else:
        _message = f"**Gaussian case:** inside the data range the largest polynomial error is **{runge_grid_error:.3g}**."
    if show_extrapolated.value:
        _message += f" Outside the data range, the largest polynomial error is **{runge_extrapolation_error:.3g}**."
    else:
        _message += " Turn on **Show extrapolated regions** to extend the plot beyond the supplied data."
    mo.vstack([runge_figure, mo.md(_message)])
    return runge_figure, runge_extrapolation_error, runge_grid_error, runge_node_error


if __name__ == "__main__":
    app.run()
