# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.15", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.differentiate import derivative

    return derivative, mo, np, plt


@app.cell
def _(np):
    def first_der(f, h):
        fp = np.empty_like(f)
        fp[1:-1] = (f[2:] - f[:-2]) / (2 * h)                         # central, interior
        fp[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)               # three-point forward
        fp[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)            # three-point backward
        return fp


    def second_der(f, h):
        fpp = np.empty_like(f)
        fpp[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / h**2             # central, interior
        fpp[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / h**2       # four-point forward
        fpp[-1] = (2 * f[-1] - 5 * f[-2] + 4 * f[-3] - f[-4]) / h**2  # four-point backward
        return fpp

    return first_der, second_der


@app.cell
def _():
    # Lennard-Jones pair in reduced units: r in sigma, V in epsilon
    def V(r):
        return 4 * (r**-12 - r**-6)


    def F_exact(r):
        return 24 * (2 * r**-13 - r**-7)


    def k_exact(r):
        return 24 * (26 * r**-14 - 7 * r**-8)

    return F_exact, V, k_exact


@app.cell
def _(mo):
    n_slider = mo.ui.slider(steps=[11, 21, 41, 81], value=21, label="samples", show_value=True)
    return (n_slider,)


@app.cell
def _(F_exact, V, first_der, k_exact, n_slider, np, second_der):
    r = np.linspace(1.0, 2.0, n_slider.value)
    h = r[1] - r[0]
    E = V(r)

    F = -first_der(E, h)
    k = second_der(E, h)
    F_np = -np.gradient(E, h, edge_order=2)

    i_max = np.argmax(np.abs(F - F_exact(r)))
    summary = f"""
    | spacing $h$ | max force error | at $r$ | max $V''$ error | `first_der` = `np.gradient` |
    |---|---|---|---|---|
    | {h:.4f} | {np.abs(F - F_exact(r))[i_max]:.3e} | {r[i_max]:.3f} | {np.max(np.abs(k - k_exact(r))):.3e} | {np.allclose(F, F_np)} |
    """
    return E, F, k, r, summary


@app.cell
def _(E, F, F_exact, k, k_exact, mo, n_slider, np, plt, r, summary):
    r_fine = np.linspace(1.0, 2.0, 400)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    axes[0].plot(r, E, "o", ms=4, color="k")
    axes[0].set(xlabel="r (σ)", ylabel="V (ε)", title="Tabulated energy", ylim=(-1.2, 1.0))
    axes[1].plot(r_fine, F_exact(r_fine), color="C0", label="exact")
    axes[1].plot(r, F, "o", ms=4, color="C3", label="-first_der")
    axes[1].set(xlabel="r (σ)", ylabel="F (ε/σ)", title="Force", ylim=(-4, 24))
    axes[2].plot(r_fine, k_exact(r_fine), color="C0", label="exact")
    axes[2].plot(r, k, "o", ms=4, color="C3", label="second_der")
    axes[2].set(xlabel="r (σ)", ylabel="V'' (ε/σ²)", title="Stiffness", ylim=(-20, 600))
    for _ax in axes:
        _ax.grid(alpha=0.25)
    axes[1].legend()
    axes[2].legend()
    mo.vstack([mo.hstack([n_slider, mo.md(summary)], justify="start", align="center", gap=2, widths=[1, 3]), fig])
    return


@app.cell
def _(F_exact, V, derivative, mo):
    res = derivative(V, 1.2)
    mo.md(
        f"""
    | `scipy.differentiate.derivative` at $r=1.2$ | value |
    |---|---|
    | force `-res.df` | {-res.df:.12f} |
    | exact force | {F_exact(1.2):.12f} |
    | reported error `res.error` | {res.error:.1e} |
    | function calls `res.nfev` | {res.nfev} |
    """
    )
    return


if __name__ == "__main__":
    app.run()
