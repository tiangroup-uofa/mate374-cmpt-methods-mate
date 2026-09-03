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
    # Interatomic potentials: implement and plot

    Three classical pair potentials that describe the energy between two
    atoms as a function of separation $r$:

    | Potential | Expression |
    |---|---|
    | **Lennard-Jones** | $V(r)=4\varepsilon\!\left[\left(\sigma/r\right)^{12}-\left(\sigma/r\right)^{6}\right]$ |
    | **Morse** | $V(r)=D_e\!\left[1-e^{-a(r-r_e)}\right]^2 - D_e$ |
    | **Sutherland** | $V(r)=-\varepsilon$ for $r\le\sigma$, $\;V(r)=-\varepsilon(\sigma/r)^{6}$ for $r>\sigma$ |

    Your job: fill in each function body, then run the cells to see all
    three curves on one plot.
    """)
    return


@app.cell
def _():
    epsilon = 0.01  # eV
    sigma = 2.55    # Angstrom
    D_e = 0.01      # eV
    a_morse = 1.4   # 1/Angstrom
    r_e = 2.87      # Angstrom (equilibrium distance for Morse)
    return D_e, a_morse, epsilon, r_e, sigma


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 1 — Lennard-Jones

    $$V_{\mathrm{LJ}}(r) = 4\varepsilon\!\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

    Replace `...` with the correct expression.
    """)
    return


@app.cell
def _():
    def lennard_jones(r, epsilon, sigma):
        # TODO: implement the Lennard-Jones potential
        return ...

    return (lennard_jones,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 2 — Morse

    $$V_{\mathrm{Morse}}(r) = D_e\!\left[1 - e^{-a(r - r_e)}\right]^2 - D_e$$

    `np.exp` gives you the exponential.
    """)
    return


@app.cell
def _():
    def morse(r, D_e, a, r_e):
        # TODO: implement the Morse potential
        return ...

    return (morse,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 3 — Sutherland

    $$V_{\mathrm{Suth}}(r) = \begin{cases} -\varepsilon & r \le \sigma \\ -\varepsilon\!\left(\sigma/r\right)^{6} & r > \sigma \end{cases}$$

    `np.where(condition, value_if_true, value_if_false)` lets you handle the two branches without a loop.
    """)
    return


@app.cell
def _():
    def sutherland(r, epsilon, sigma):
        # TODO: implement the Sutherland potential
        # hint: np.where(r <= sigma, ..., ...)
        return ...

    return (sutherland,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 4 — plot all three

    The cell below builds the $r$ array and plots the three potentials.
    Once your functions return numbers, the figure will appear.
    """)
    return


@app.cell
def _(
    D_e,
    a_morse,
    epsilon,
    lennard_jones,
    mo,
    morse,
    np,
    plt,
    r_e,
    sigma,
    sutherland,
):
    r = np.linspace(2.0, 6.0, 500)

    V_lj = lennard_jones(r, epsilon, sigma)
    V_morse = morse(r, D_e, a_morse, r_e)
    V_suth = sutherland(r, epsilon, sigma)

    _has_data = (
        V_lj is not None
        and V_morse is not None
        and V_suth is not None
        and not isinstance(V_lj, type(Ellipsis))
        and not isinstance(V_morse, type(Ellipsis))
        and not isinstance(V_suth, type(Ellipsis))
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    if _has_data:
        ax.plot(r, V_lj, label="Lennard-Jones")
        ax.plot(r, V_morse, label="Morse")
        ax.plot(r, V_suth, label="Sutherland")
        ax.set_ylim(-0.015, 0.02)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$V(r)$ (eV)")
    ax.set_title("Pair potentials")
    ax.legend()
    fig.tight_layout()

    if _has_data:
        mo.callout(
            mo.md("All three potentials are plotted. Compare the well depth, equilibrium distance, and repulsive wall."),
            kind="success",
        )
    else:
        mo.callout(
            mo.md("Fill in the function bodies above, then run those cells."),
            kind="warn",
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Stretch goals

    1. Find the equilibrium distance (minimum of $V$) for Lennard-Jones from the plot. Verify against $r_{\min} = 2^{1/6}\sigma$.
    2. Add a slider for $\varepsilon$ or $\sigma$ and see how the curves shift.
    3. Plot the force $F(r) = -dV/dr$ numerically using `np.gradient(V, r)`.
    """)
    return


if __name__ == "__main__":
    app.run()
