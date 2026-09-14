# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.9", "numpy>=2.0", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import root_scalar
    return mo, np, plt, root_scalar


@app.cell(hide_code=True)
def _(mo):
    method = mo.ui.dropdown(
        ["bisect", "brentq", "brenth", "ridder", "toms748"],
        value="bisect", label="SciPy method",
    )
    mo.vstack([
        mo.md("**CO₂ · 1 mol · 280 K · 50 bar**\n\n"
              "Bracket: [0.2, 0.6] L · xtol: 10⁻⁶ L · maxiter: 100"),
        method,
    ])
    return (method,)


@app.cell(hide_code=True)
def _():
    def F(V):
        n, R, T = 1.0, 0.08314462618, 280.0
        a_gas, b_gas, P_target = 3.592, 0.04267, 50.0
        return n * R * T / (V - n * b_gas) - a_gas * n**2 / V**2 - P_target
    return (F,)


@app.cell(hide_code=True)
def _(F, method, np, root_scalar):
    def solve_with_trace(method_name):
        evaluations = []

        def recorded_F(V):
            value = F(V)
            evaluations.append((V, value))
            return value

        solution = root_scalar(
            recorded_F, bracket=[0.2, 0.6], method=method_name,
            xtol=1e-6, rtol=1e-12, maxiter=100,
        )
        return solution, np.array(evaluations)

    result, samples = solve_with_trace(method.value)
    return result, samples


@app.cell(hide_code=True)
def _(F, mo, np, plt, result, samples):
    volumes = np.linspace(0.2, 0.6, 500)
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.plot(volumes, F(volumes), color="#007c41", label="F(V)")
    ax.axhline(0, color="#444444", lw=1)
    dots = ax.scatter(
        samples[:, 0], samples[:, 1], c=np.arange(1, len(samples) + 1),
        cmap="plasma", s=45, edgecolor="#444444", linewidth=0.4,
        label="Function evaluations", zorder=3,
    )
    fig.colorbar(dots, ax=ax, label="Evaluation order", ticks=[1, len(samples)])
    ax.set(xlabel="Volume (L)", ylabel="Residual (bar)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    mo.vstack([
        fig,
        mo.md(f"""
        Volume: **{result.root:.7f} L** · residual: **{F(result.root):+.3g} bar**

        Function evaluations: **{result.function_calls}** ·
        iterations: **{result.iterations}** · converged: **{result.converged}**

        Dots show function evaluations, including the bracket endpoints.
        Which method needs the fewest?
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
