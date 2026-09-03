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
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Implement a function and plot it

    **Task.** Write the thermal-expansion function

    $$
    \Delta L = \alpha \, L_0 \, \Delta T
    $$

    then plot $\Delta L$ against $\Delta T$ for a range of temperature changes.
    """)
    return


@app.cell
def _():
    L0_mm = 100.0
    alpha_per_K = 23.0e-6
    return L0_mm, alpha_per_K


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 1 — fill in the function body

    Replace `...` with the correct expression and run the cell.
    """)
    return


@app.cell
def _():
    def thermal_expansion(L0: float, alpha: float, dT):
        # TODO: return the predicted length change in mm
        return ...

    return (thermal_expansion,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 2 — build an array of temperature changes and evaluate
    """)
    return


@app.cell
def _(L0_mm, alpha_per_K, np, thermal_expansion):
    dT = np.linspace(-100, 300, 200)
    dL = thermal_expansion(L0_mm, alpha_per_K, dT)
    return dL, dT


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 3 — plot the result

    The cell below plots $\Delta L$ vs $\Delta T$. Once your function
    returns real numbers instead of `...`, the figure will appear.
    """)
    return


@app.cell
def _(dL, dT, mo, plt):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(dT, dL)
    ax.set_xlabel(r"$\Delta T$ (K)")
    ax.set_ylabel(r"$\Delta L$ (mm)")
    ax.set_title("Thermal expansion")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()

    mo.callout(
        mo.md("Plot is working — your function returned numerical values."),
        kind="success",
    ) if dL is not None and not isinstance(dL, type(Ellipsis)) else mo.callout(
        mo.md("Fill in the function body above, then run that cell."),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Stretch goals

    1. Add a second material (e.g. copper, $\alpha = 17 \times 10^{-6}\,\mathrm{K^{-1}}$) to the same plot.
    2. Add a legend with `ax.legend()`.
    3. Find the $\Delta T$ where $\Delta L = 0.5\,\mathrm{mm}$ by reading the plot or by algebra.
    """)
    return


if __name__ == "__main__":
    app.run()
