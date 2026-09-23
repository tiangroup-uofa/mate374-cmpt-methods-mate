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


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # A1 playground: regular-solution free energy

    For a binary regular solution,

    $$
    \Delta G_{\mathrm{mix}}(x,T)
    =\Omega x(1-x)
    +RT\left[x\ln x+(1-x)\ln(1-x)\right].
    $$

    This playground evaluates the equation over a **one-dimensional
    composition array** at one temperature. A1 asks you to extend the
    same NumPy expression to all combinations of composition and
    temperature and display the result as a two-dimensional map.
    """)
    return


@app.cell
def _(mo):
    temperature_K = mo.ui.slider(
        start=300.0,
        stop=1500.0,
        step=25.0,
        value=300.0,
        label="Temperature, T (K)",
        show_value=True,
    )
    temperature_K
    return (temperature_K,)


@app.cell
def _(mo, np, plt, temperature_K):
    def _plot_regular_solution(selected_temperature_K):
        interaction_kJ_per_mol = 20.0
        gas_constant_kJ_per_mol_K = 8.314e-3
        composition = np.linspace(0.01, 0.99, 200)

        free_energy_kJ_per_mol = (
            interaction_kJ_per_mol * composition * (1.0 - composition)
            + gas_constant_kJ_per_mol_K
            * selected_temperature_K
            * (
                composition * np.log(composition)
                + (1.0 - composition) * np.log(1.0 - composition)
            )
        )

        midpoint_energy = (
            interaction_kJ_per_mol * 0.5 * (1.0 - 0.5)
            + gas_constant_kJ_per_mol_K
            * selected_temperature_K
            * (0.5 * np.log(0.5) + 0.5 * np.log(0.5))
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(composition, free_energy_kJ_per_mol, linewidth=2)
        ax.scatter([0.5], [midpoint_energy], color="#d95f02", zorder=3)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xlabel("Mole fraction of component B, x")
        ax.set_ylabel(r"Free energy of mixing, $\Delta G_{mix}$ (kJ mol$^{-1}$)")
        ax.set_title(f"Regular solution at T = {selected_temperature_K:.0f} K")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        check = mo.callout(
            mo.md(
                fr"""
                At $x=0.5$ and $T={selected_temperature_K:.0f}\,\mathrm{{K}}$,
                the calculated value is
                **{midpoint_energy:.4f} kJ mol⁻¹**.

                Use the assigned $x=0.5$, $T=300\,\mathrm{{K}}$ value as a
                checkpoint before constructing the two-dimensional grid.
                """
            ),
            kind="info",
        )
        return mo.vstack([check, fig], gap=1.5)

    _plot_regular_solution(temperature_K.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Transfer to A1

    Open **Edit code** and locate the array expression. In A1, create a
    composition-temperature grid with NumPy broadcasting or
    `np.meshgrid`, evaluate the equation on that grid, and label both
    axes and the colour scale. The playground intentionally does not
    construct the required two-dimensional answer.
    """)
    return


if __name__ == "__main__":
    app.run()
