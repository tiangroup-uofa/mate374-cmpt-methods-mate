# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    rule = mo.ui.dropdown(
        options=["left rectangle", "midpoint", "trapezoid", "Simpson 1/3", "Simpson 3/8"],
        value="trapezoid",
        label="rule",
    )
    n_sub = mo.ui.slider(steps=[1, 2, 3, 4, 6, 12], value=2, label="subintervals N", show_value=True)
    return n_sub, rule


@app.cell
def _(np):
    def composite(rule, f, a, b, n):
        """Composite Newton-Cotes estimate of the integral of f from a to b with n subintervals."""
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        if rule == "left rectangle":
            return h * np.sum(f(x[:-1]))
        if rule == "midpoint":
            return h * np.sum(f((x[:-1] + x[1:]) / 2))
        fx = f(x)
        if rule == "trapezoid":
            return h * (fx[0] / 2 + np.sum(fx[1:-1]) + fx[-1] / 2)
        if rule == "Simpson 1/3":
            return h / 3 * (fx[0] + 4 * np.sum(fx[1:-1:2]) + 2 * np.sum(fx[2:-1:2]) + fx[-1])
        if rule == "Simpson 3/8":
            w = np.full(n + 1, 3.0)
            w[0] = w[-1] = 1.0
            w[3:-1:3] = 2.0
            return 3 * h / 8 * np.sum(w * fx)

    return (composite,)


@app.cell
def _(composite, mo, n_sub, np, plt, rule):
    def f(x):
        return 1.0 + 1.5 * np.sin(0.62 * x + 0.15) + 0.1 * x


    a, b = 0.4, 3.9
    exact = (b - a) - 1.5 / 0.62 * (np.cos(0.62 * b + 0.15) - np.cos(0.62 * a + 0.15)) + 0.05 * (b**2 - a**2)

    n = n_sub.value
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    width = {"Simpson 1/3": 2, "Simpson 3/8": 3}.get(rule.value, 1)   # subintervals per panel

    if n % width != 0:
        _need = "an even number" if width == 2 else "a multiple of 3"
        output = mo.callout(mo.md(f"**{rule.value}** needs {_need} of subintervals. Choose another N."), kind="warn")
    else:
        estimate = composite(rule.value, f, a, b, n)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.68)
        used = []
        for _k in range(0, n, width):
            _x0, _x1 = x[_k], x[_k + width]
            _xs = np.linspace(_x0, _x1, 60)
            if rule.value == "left rectangle":
                _nodes = np.array([_x0])
            elif rule.value == "midpoint":
                _nodes = np.array([(_x0 + _x1) / 2])
            else:
                _nodes = np.linspace(_x0, _x1, width + 1)
            if _nodes.size == 1:
                _p = np.full_like(_xs, f(_nodes[0]))
            else:
                _p = np.polyval(np.polyfit(_nodes, f(_nodes), _nodes.size - 1), _xs)
            ax.fill_between(_xs, 0, _p, color="C2", alpha=0.3, lw=0)
            ax.plot(_xs, _p, color="C3", lw=1.5, ls="--")
            ax.plot(_nodes, f(_nodes), "o", color="k", ms=5)
            used.extend(np.round(_nodes, 12))
        _xf = np.linspace(0.1, 4.25, 300)
        ax.plot(_xf, f(_xf), color="k", lw=1.6)
        ax.set(xlim=(0.1, 4.25), ylim=(0, 3.1))
        ax.set_xlabel("x", fontsize=15)
        ax.set_ylabel("f(x)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.set_title(f"{rule.value}, N = {n}, h = {h:.3f}", fontsize=15)
        ax.grid(alpha=0.25)
        fig.text(
            0.5, 0.96,
            r"$f(x)=1+1.5\sin(0.62x+0.15)+0.1x,\quad x\in[0.4,\,3.9]$",
            ha="center", va="top", fontsize=17,
        )
        fig.text(
            0.5, 0.87,
            f"Exact = {exact:.6f}     Estimate = {estimate:.6f}",
            ha="center", va="top", fontsize=17,
        )
        fig.text(
            0.5, 0.79,
            f"Error (exact − estimate) = {exact - estimate:+.2e}     Values used: {len(set(used))}",
            ha="center", va="top", fontsize=15,
        )
        output = fig
        plt.close(fig)

    mo.vstack([mo.hstack([rule, n_sub], justify="start", gap=2), output])
    return


if __name__ == "__main__":
    app.run()
