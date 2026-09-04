# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
# ]
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
    mo.md(r"""
    # From equation to code to plot

    The Lennard-Jones potential models how the interaction energy $V$
    between two atoms changes with their separation $r$:

    $$
    V(r)=4\varepsilon\left[
    \left(\frac{\sigma}{r}\right)^{12}
    -
    \left(\frac{\sigma}{r}\right)^6
    \right].
    $$

    Here, $\varepsilon$ sets the depth of the attractive well and
    $\sigma$ is the separation where $V=0$.
    """)
    return


@app.cell
def _(np):
    distance_A = np.linspace(3.2, 8.0, 400)  # Angstrom
    # Representative Lennard-Jones parameters for argon
    epsilon_eV = 0.0103  # eV
    sigma_A = 3.40  # Angstrom
    return distance_A, epsilon_eV, sigma_A


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your turn

    In the code cell below, replace only the `...` with the
    Lennard-Jones expression. In Python, write a power such as
    $(\sigma/r)^{12}$ as `(sigma / r) ** 12`.

    Once `LJ_potential` is implemented correctly, the plot will appear
    automatically.
    """)
    return


@app.cell
def _(np):
    def LJ_potential(
        r: np.ndarray,
        epsilon: float,
        sigma: float,
    ) -> np.ndarray:
        return ...

    return (LJ_potential,)


@app.cell(hide_code=True)
def _(LJ_potential, distance_A, epsilon_eV, mo, plt, sigma_A):
    potential_eV = LJ_potential(distance_A, epsilon_eV, sigma_A)

    if potential_eV is ...:
        plot_output = mo.callout(
            mo.md("Replace `...` in `LJ_potential`, then run that cell."),
            kind="warn",
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(distance_A, potential_eV, color="#275D38", linewidth=2)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xlabel(r"Separation, $r$ ($\mathrm{\AA}$)")
        ax.set_ylabel(r"Potential energy, $V(r)$ (eV)")
        ax.set_title("Lennard-Jones pair potential")
        ax.set_ylim(-0.012, 0.03)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plot_output = fig

    plot_output
    return


if __name__ == "__main__":
    app.run()
