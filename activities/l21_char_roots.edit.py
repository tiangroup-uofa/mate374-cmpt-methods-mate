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
    # a y'' + b y' + c y = 0, listed as (a, b, c)
    CASES = {
        "b = 0: harmonic bond, m = 1, k = 4": (1.0, 0.0, 4.0),
        "b² > 4ac: two real roots (overdamped)": (1.0, 3.0, 2.0),
        "b² < 4ac: complex roots (damped oscillation)": (1.0, 0.6, 4.0),
        "c = 0, b = 0: double root (steady diffusion)": (1.0, 0.0, 0.0),
    }
    case = mo.ui.dropdown(options=list(CASES), value="b = 0: harmonic bond, m = 1, k = 4", label="equation")
    y0 = mo.ui.slider(steps=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], value=1.0, label="y(0)", show_value=True)
    v0 = mo.ui.slider(steps=[-2.0, -1.0, 0.0, 1.0, 2.0], value=0.0, label="y'(0)", show_value=True)
    return CASES, case, v0, y0


@app.cell
def _(CASES, case, mo, np, plt, v0, y0):
    a, b, c = CASES[case.value]
    disc = b * b - 4 * a * c
    roots = np.roots([a, b, c])
    t = np.linspace(0, 6, 601)

    if disc > 0:
        # two real roots: y(0) = C1 + C2 and y'(0) = l1 C1 + l2 C2 form a 2 x 2 linear system
        l1, l2 = roots
        C1, C2 = np.linalg.solve([[1, 1], [l1, l2]], [y0.value, v0.value])
        y = C1 * np.exp(l1 * t) + C2 * np.exp(l2 * t)
        form = f"y = {C1:.3f} e^({l1:.3f} t) + {C2:.3f} e^({l2:.3f} t)"
        roots_text = f"λ = {l1:.3f}, {l2:.3f}"
    elif disc == 0:
        # double root: y = (C1 + C2 t) e^(l t)
        l1 = roots[0].real
        C1, C2 = y0.value, v0.value - l1 * y0.value
        y = (C1 + C2 * t) * np.exp(l1 * t)
        form = f"y = ({C1:.3f} + {C2:.3f} t) e^({l1:.3f} t)"
        roots_text = f"λ = {l1:.3f} (double)"
    else:
        # complex pair p ± i q: y = e^(p t) (A cos q t + B sin q t)
        p, q = roots[0].real, abs(roots[0].imag)
        A, B = y0.value, (v0.value - p * y0.value) / q
        y = np.exp(p * t) * (A * np.cos(q * t) + B * np.sin(q * t))
        form = f"y = e^({p:.3f} t) [{A:.3f} cos({q:.3f} t) + {B:.3f} sin({q:.3f} t)]"
        roots_text = f"λ = {p:.3f} ± {q:.3f} i"

    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    ax.plot(t, y, color="C0", lw=2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="t", ylabel="y(t)")
    ax.grid(alpha=0.25)

    mo.vstack(
        [
            mo.hstack([case, y0, v0], justify="start", align="center", gap=2),
            mo.md(f"**Characteristic roots:** {roots_text}  \n**Solution:** `{form}`"),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

