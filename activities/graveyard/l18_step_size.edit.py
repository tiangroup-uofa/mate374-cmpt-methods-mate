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
    mo.md(r"""
    ## How small should the displacement be?
    Predict where forward and central differences of $f(x)=x^3$ at $x=3$
    will stop improving. The exact slope is 27. Here $x$ and $f$ are dimensionless.
    Edit `dtype` to compare 64-bit and 32-bit arithmetic.
    """)
    return


@app.cell
def _(np):
    dtype = np.float64                 # Try np.float32
    x = dtype(3.0)
    h = np.logspace(-16, 0, 321, dtype=dtype)

    def f(t):
        return t**3

    forward = (f(x + h) - f(x)) / h
    central = (f(x + h) - f(x - h)) / (dtype(2) * h)
    exact = dtype(27.0)
    print("Machine epsilon:", np.finfo(dtype).eps)
    print("Spacing at x = 3:", np.spacing(x))
    return central, dtype, exact, f, forward, h, x


@app.cell
def _(central, exact, forward, h, np, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for _name, _estimate in [("Forward", forward), ("Central", central)]:
        axes[0].semilogx(h, _estimate, label=_name)
        _error = np.abs(_estimate - exact)
        # Exact floating-point matches cannot be placed on a logarithmic axis.
        axes[1].loglog(h, np.where(_error > 0, _error, np.nan), label=_name)
        print(f"{_name}: {np.count_nonzero(_error == 0)} exact floating-point matches (omitted from error plot)")
    axes[0].axhline(exact, color="black", ls=":", label="Exact = 27")
    axes[0].set(xlabel="Step h", ylabel="Derivative estimate", ylim=(0, 40))
    axes[1].loglog(h, 9 * h + h**2, ":", color="C0", label="Forward truncation: 9h + h²")
    axes[1].loglog(h, h**2, ":", color="C1", label="Central truncation: h²")
    axes[1].set(xlabel="Step h", ylabel="Absolute derivative error", ylim=(1e-16, 1e2))
    for _ax in axes:
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.25)
    fig
    return


@app.cell
def _(dtype, f, x):
    print("       h          forward          central")
    for _value in [1e-1, 1e-3, 1e-5, 1e-8, 1e-13, 1e-16]:
        _h = dtype(_value)
        _forward = (f(x + _h) - f(x)) / _h
        _central = (f(x + _h) - f(x - _h)) / (dtype(2) * _h)
        print(f"{_h:10.1e}  {_forward:15.9f}  {_central:15.9f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Interpretation.** Read the plots from right to left as $h$ decreases.
    The truncation curves predict the initial improvement, but the computed
    errors eventually become irregular. Look for a range of accurate steps,
    rather than one accidentally exact point. In an energy calculation, the
    same subtraction determines how well a displacement resolves the force.
    Why does the useful range move when you select `np.float32`?
    """)
    return


if __name__ == "__main__":
    app.run()
