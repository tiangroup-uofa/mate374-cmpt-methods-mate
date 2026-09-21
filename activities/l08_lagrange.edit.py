# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.0.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def data(np):
    # Seven data points for the interpolation experiment.
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_data = np.array([7.0, 11.0, 28.0, 24.0, 19.0, 30.0, 27.0])
    return x_data, y_data


@app.cell(hide_code=True)
def controls(mo):
    point_count = mo.ui.slider(
        2,
        7,
        value=7,
        step=1,
        label="Number of data points included",
        show_value=True,
    )
    point_count
    return (point_count,)


@app.cell(hide_code=True)
def lagrange(np):
    def lagrange_basis(x, i, x_data):
        numerator = 1.0
        denominator = 1.0
        x_i = x_data[i]
        for j, x_j in enumerate(x_data):
            if i == j:
                continue
            numerator *= x - x_j
            denominator *= x_i - x_j
        return numerator / denominator

    def lagrange_interpolation(x, x_data, y_data):
        total = 0.0
        for i, y_i in enumerate(y_data):
            total += y_i * lagrange_basis(x, i, x_data)
        return total

    return lagrange_basis, lagrange_interpolation


@app.cell(hide_code=True)
def plot(lagrange_basis, lagrange_interpolation, mo, np, plt, point_count, x_data, y_data):
    n_points = point_count.value
    selected_x = x_data[:n_points]
    selected_y = y_data[:n_points]
    x_grid = np.linspace(selected_x[0], selected_x[-1], 800)

    lagrange_figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

    for i in range(n_points):
        axes[0].plot(x_grid, lagrange_basis(x_grid, i, selected_x), lw=1.8, label=rf"$L_{i}(x)$")
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].axhline(1, color="0.5", lw=0.8, ls=":")
    axes[0].set(
        xlabel="x",
        ylabel="basis value",
        title=f"The {n_points} selected Lagrange basis polynomials",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)

    interpolation = lagrange_interpolation(x_grid, selected_x, selected_y)
    axes[1].plot(x_grid, interpolation, color="black", lw=2.5, label=rf"$P_{{{n_points - 1}}}(x)$")
    axes[1].scatter(selected_x, selected_y, color="black", s=42, zorder=4, label="selected data")
    axes[1].set(
        xlabel="x",
        ylabel="y",
        title="Interpolating polynomial",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    lagrange_figure
    return lagrange_figure, n_points, selected_x, selected_y


if __name__ == "__main__":
    app.run()
