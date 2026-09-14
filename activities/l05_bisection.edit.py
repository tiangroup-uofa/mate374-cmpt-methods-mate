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
    method = mo.ui.dropdown(
        ["Bisection", "False position"], value="Bisection", label="Method",
    )
    steps = mo.ui.number(1, 20, step=1, value=1, label="Step")
    bracket = mo.ui.range_slider(
        0.05, 0.8, step=0.01, value=[0.2, 0.6],
        label="Initial bracket (L)", show_value=True, debounce=True,
    )
    mo.vstack([
        mo.md("**CO₂ · 1 mol · 280 K · 50 bar.** Which subinterval will we keep?"),
        mo.hstack([method, steps], widths="equal"),
        bracket,
    ])
    return bracket, method, steps


@app.cell(hide_code=True)
def _():
    def F(V):
        n, R, T = 1.0, 0.08314462618, 280.0
        a_gas, b_gas, P_target = 3.592, 0.04267, 50.0
        return n * R * T / (V - n * b_gas) - a_gas * n**2 / V**2 - P_target

    def trace_steps(f, left, right, count, method):
        # Begin with a strict sign-changing bracket.
        history = []
        for step in range(count):
            fa, fb = f(left), f(right)
            if method == "Bisection":
                trial = (left + right) / 2
            else:
                trial = (left * fb - right * fa) / (fb - fa)
            ft = f(trial)
            old_left, old_right = left, right
            if ft == 0:
                left = right = trial
            elif fa * ft < 0:
                right = trial
            else:
                left = trial
            history.append((old_left, old_right, trial, ft, left, right))
            if ft == 0:
                break
        return history
    return F, trace_steps


@app.cell(hide_code=True)
def _(F, bracket, method, mo, steps, trace_steps):
    initial_left, initial_right = bracket.value
    mo.stop(
        F(initial_left) == 0 or F(initial_right) == 0,
        mo.md("An endpoint is already a root. Choose another bracket to trace the steps."),
    )
    mo.stop(
        initial_left >= initial_right or F(initial_left) * F(initial_right) > 0,
        mo.md("Choose distinct endpoints with opposite signs. Try [0.20, 0.60] L."),
    )
    history = trace_steps(F, initial_left, initial_right, steps.value, method.value)
    return history, initial_left, initial_right


@app.cell(hide_code=True)
def _(F, history, initial_left, initial_right, method, mo, np, plt):
    old_left, old_right, trial, residual, left, right = history[-1]
    volumes = np.linspace(initial_left, initial_right, 1000)
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.plot(volumes, F(volumes), color="#007c41", label="F(V)")
    ax.axhline(0, color="#444444", lw=1)
    ax.axvspan(left, right, alpha=0.18, color="#007c41", label="Retained bracket")
    ax.scatter([old_left, old_right], [F(old_left), F(old_right)],
               color="#444444", s=35, label="Current endpoints", zorder=3)
    if method.value == "False position":
        ax.plot([old_left, old_right], [F(old_left), F(old_right)],
                color="#d87700", ls="--", label="Straight-line estimate")
    ax.scatter([trial], [residual], color="#d87700", s=70, zorder=4, label="Trial point")
    ax.axvline(trial, color="#d87700", ls=":")
    ax.set(xlabel="Volume (L)", ylabel="Residual (bar)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    mo.vstack([
        fig,
        mo.md(f"Step **{len(history)}** · trial **{trial:.7f} L** · "
              f"F = **{residual:+.4g} bar**\n\n"
              f"Retained bracket: **[{left:.7f}, {right:.7f}] L** · "
              f"width: **{right - left:.3g} L**"),
    ])
    return


if __name__ == "__main__":
    app.run()
