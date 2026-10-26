# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import integrate

    return integrate, mo, np, plt


@app.cell
def _(mo):
    points = mo.ui.dropdown(options=["5", "9", "17"], value="5", label="measured radii")
    return (points,)


@app.cell
def _(integrate, mo, np, points):
    R = 10.0          # pipe radius (mm)
    u_max = 2.0       # centre-line velocity (m/s)
    rho = 900.0       # quench-oil density (kg/m³)

    r = np.linspace(0.0, R, int(points.value))     # radii (mm), evenly spaced
    u = u_max * (1 - (r / R) ** 2)                   # measured velocities (m/s)
    f = u * r                                        # integrand u r (mm·m/s)
    h = r[1] - r[0]

    I_trap = np.trapezoid(f, r)                      # mm²·m/s
    I_simp = integrate.simpson(f, x=r)
    I_romb = integrate.romb(f, dx=h)                 # needs 2**k + 1 evenly spaced samples
    I_exact = u_max * R**2 / 4

    _lines = ["| method | ∫ u r dr (mm²·m/s) | ṁ (kg/s) | error |", "|---|---|---|---|"]
    for _name, _I in [("np.trapezoid", I_trap), ("integrate.simpson", I_simp), ("integrate.romb", I_romb), ("exact", I_exact)]:
        _mdot = 2 * np.pi * rho * _I * 1e-6          # mm²·m/s -> m³/s, then kg/s
        _lines.append(f"| {_name} | {_I:.3f} | {_mdot:.4f} | {100 * (_I - I_exact) / I_exact:+.2f} % |")
    mo.vstack([points, mo.md("\n".join(_lines))])
    return R, f, r, u_max


@app.cell
def _(R, f, np, plt, r, u_max):
    _rf = np.linspace(0, R, 200)
    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    ax.plot(_rf, u_max * (1 - (_rf / R) ** 2) * _rf, color="0.6", lw=2, label="exact u r")
    ax.plot(r, f, "o-", color="C3", ms=5, lw=1, label="samples joined by trapezoids")
    ax.set(xlabel="r (mm)", ylabel="u r (mm·m/s)", title=f"{r.size} radii, h = {r[1] - r[0]:g} mm")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig
    return


if __name__ == "__main__":
    app.run()
