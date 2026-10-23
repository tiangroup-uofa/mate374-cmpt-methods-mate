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
    offset_choice = mo.ui.dropdown(options=["0", "-1e4", "-1e8", "-1e12"], value="-1e8", label="energy offset E0 (eV)")
    h_choice = mo.ui.dropdown(options=["1e-3", "1e-4", "1e-5"], value="1e-5", label="h (Å)")
    return h_choice, offset_choice


@app.cell
def _(h_choice, np, offset_choice):
    E0 = float(offset_choice.value)
    h = float(h_choice.value)
    k = 1.0                                  # eV/Å²


    def energy(q):
        return E0 + 0.5 * k * q**2


    q = 1e-4                                 # Å
    E_plus, E_minus = energy(q + h), energy(q - h)
    F_central = (E_minus - E_plus) / (2 * h)

    summary = f"""
    | at $q$ = 1e-4 Å | value |
    |---|---|
    | float64 spacing near $E_0$ | {abs(np.spacing(E0)):.3e} eV |
    | $E(q+h)$ | `{E_plus!r}` |
    | $E(q-h)$ | `{E_minus!r}` |
    | stored difference | {E_plus - E_minus:.3e} eV |
    | exact difference | {2 * h * k * q:.3e} eV |
    | central force | {F_central:.4e} eV/Å |
    | true force | {-k * q:.4e} eV/Å |
    """
    return energy, h, k, summary


@app.cell
def _(energy, h, h_choice, k, mo, np, offset_choice, plt, summary):
    q_values = np.linspace(-5e-4, 5e-4, 201)
    F_values = -(energy(q_values + h) - energy(q_values - h)) / (2 * h)

    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    ax.plot(q_values * 1e4, -k * q_values * 1e4, color="0.6", label="true force  −kq")
    ax.plot(q_values * 1e4, F_values * 1e4, ".", ms=3, color="C3", label="central difference")
    ax.set(xlabel="displacement q (10⁻⁴ Å)", ylabel="force (10⁻⁴ eV/Å)", ylim=(-15, 15))
    ax.grid(alpha=0.25)
    ax.legend()
    mo.vstack(
        [
            mo.hstack([mo.vstack([offset_choice, h_choice]), mo.md(summary)], justify="start", align="center", gap=2, widths=[2, 3]),
            fig,
        ]
    )
    return



if __name__ == "__main__":
    app.run()
