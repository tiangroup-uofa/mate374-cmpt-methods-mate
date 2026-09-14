# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.9", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    gas = mo.ui.dropdown(
        ["Carbon dioxide", "Nitrogen", "Methane"],
        value="Carbon dioxide", label="Gas",
    )
    model = mo.ui.dropdown(
        ["Ideal gas", "van der Waals"], value="Ideal gas", label="EOS",
    )
    temperature = mo.ui.number(80, 400, step=5, value=280, label="T (K)")
    target = mo.ui.number(1, 100, step=1, value=50, label="Target P (bar)")
    interval = mo.ui.range_slider(
        0.05, 5.0, step=0.01, value=[0.05, 0.80],
        label="Search interval (L)", show_value=True, debounce=True,
    )
    grid_size = mo.ui.slider(
        steps=[5, 10, 20, 50, 100, 200, 500], value=20,
        label="Grid intervals, N", show_value=True, debounce=True,
    )
    mo.vstack([
        mo.hstack([gas, model], justify="start"),
        mo.hstack([temperature, target], justify="start"),
        mo.hstack([interval, grid_size], widths="equal"),
    ])
    return gas, grid_size, interval, model, target, temperature


@app.cell(hide_code=True)
def _(gas, model, target, temperature):
    # a: L² bar / mol²; b: L / mol. Rounded teaching parameters.
    parameters = {
        "Carbon dioxide": (3.592, 0.04267),
        "Nitrogen": (1.370, 0.03870),
        "Methane": (2.303, 0.04310),
    }
    a_gas, b_gas = parameters[gas.value]
    n = 1.0  # mol
    R = 0.08314462618  # L bar / (mol K)
    T = temperature.value
    P_target = target.value

    def pressure(V):
        if model.value == "Ideal gas":
            return n * R * T / V
        return n * R * T / (V - n * b_gas) - a_gas * n**2 / V**2

    def F(V):
        return pressure(V) - P_target
    return F, P_target, a_gas, b_gas, pressure


@app.cell(hide_code=True)
def _(F, grid_size, interval, mo, np):
    V_left, V_right = interval.value
    mo.stop(V_left == V_right, mo.md("Choose an interval with two distinct endpoints."))
    N = grid_size.value
    V_grid = np.linspace(V_left, V_right, N + 1)
    F_grid = F(V_grid)
    V_estimate = V_grid[np.argmin(np.abs(F_grid))]
    return N, V_estimate, V_grid, V_left, V_right


@app.cell(hide_code=True)
def _(F, N, P_target, V_estimate, V_grid, V_left, V_right, a_gas, b_gas, mo, model, np, plt, pressure):
    volumes = np.linspace(V_left, V_right, 1500)
    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    ax.plot(volumes, pressure(volumes), color="#007c41", label=model.value)
    ax.axhline(P_target, color="#444444", ls="--", label="Target pressure")
    ax.scatter(V_grid, pressure(V_grid), s=18, color="#007c41", zorder=3)
    ax.scatter([V_estimate], [pressure(V_estimate)], s=90, marker="x",
               color="#d87700", label="Closest grid point", zorder=4)
    ax.set(xlabel="Volume (L)", ylabel="Pressure (bar)",
           ylim=(-20, max(100, 2 * P_target)))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    mo.vstack([
        mo.md(f"n = 1 mol · a = {a_gas:g} L² bar/mol² · b = {b_gas:g} L/mol"),
        fig,
        mo.md(f"Closest sample: **{V_estimate:.5f} L** · "
              f"residual: **{F(V_estimate):+.3f} bar** · "
              f"spacing: **{(V_right - V_left) / N:.5f} L**"),
        mo.md("How many intersections do you see? Does the grid resolve each one?"),
    ])
    return


if __name__ == "__main__":
    app.run()
