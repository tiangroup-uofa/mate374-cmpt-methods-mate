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

    Three classical pair potentials describe the energy between two
    atoms as a function of separation $r$:

    | Potential | Expression |
    |---|---|
    | **Lennard-Jones** | $V(r)=4\varepsilon\!\left[\left(\sigma/r\right)^{12}-\left(\sigma/r\right)^{6}\right]$ |
    | **Morse** | $V(r)=D_e\!\left[1-e^{-a(r-r_e)}\right]^2 - D_e$ |
    | **Sutherland** | $V(r)=-\varepsilon$ for $r\le\sigma$, $\;V(r)=-\varepsilon(\sigma/r)^{6}$ for $r>\sigma$ |

    Your job: fill in each function body below (replace the `...`),
    then run the cells. The plot appears automatically once the
    functions return numbers.
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

    Replace the `...` after `return` with the expression above.
    The inputs `r`, `epsilon`, and `sigma` are already provided as
    function arguments — use them directly.

    For example, $(\sigma/r)^{6}$ is written as `(sigma / r) ** 6`.
    """)
    return


@app.cell
def _():
    def lennard_jones(r, epsilon, sigma):
        return ...

    return (lennard_jones,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 2 — Morse

    $$V_{\mathrm{Morse}}(r) = D_e\!\left[1 - e^{-a(r - r_e)}\right]^2 - D_e$$

    Replace the `...` after `return`.
    Use `np.exp(x)` for $e^{x}$. For example, $e^{-a(r-r_e)}$ is
    `np.exp(-a * (r - r_e))`.
    """)
    return


@app.cell
def _():
    def morse(r, D_e, a, r_e):
        return ...

    return (morse,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 3 — Sutherland

    $$V_{\mathrm{Suth}}(r) = \begin{cases} -\varepsilon & r \le \sigma \\ -\varepsilon\!\left(\sigma/r\right)^{6} & r > \sigma \end{cases}$$

    This potential has two pieces. Use `np.where` to pick between them
    without writing a loop:

    ```python
    np.where(r <= sigma, value_if_true, value_if_false)
    ```

    Replace `value_if_true` with the expression for $r \le \sigma$
    and `value_if_false` with the expression for $r > \sigma$.
    """)
    return


@app.cell
def _():
    def sutherland(r, epsilon, sigma):
        return ...

    return (sutherland,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 4 — plot all three

    You do not need to edit the cell below. Once your three functions
    return numbers instead of `...`, the plot will appear.
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

    _ready = True
    _results = {}
    for _name, _fn, _args in [
        ("Lennard-Jones", lennard_jones, (r, epsilon, sigma)),
        ("Morse", morse, (r, D_e, a_morse, r_e)),
        ("Sutherland", sutherland, (r, epsilon, sigma)),
    ]:
        try:
            _v = _fn(*_args)
            if _v is ... or _v is None:
                _ready = False
            else:
                _results[_name] = np.asarray(_v, dtype=float)
        except Exception:
            _ready = False

    fig, ax = plt.subplots(figsize=(7, 4))
    if _ready and len(_results) == 3:
        for _name, _v in _results.items():
            ax.plot(r, _v, label=_name)
        ax.set_ylim(-0.015, 0.02)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$V(r)$ (eV)")
    ax.set_title("Pair potentials")
    ax.legend()
    fig.tight_layout()

    if _ready and len(_results) == 3:
        mo.callout(
            mo.md(
                "All three potentials are plotted. Compare the well "
                "depth, equilibrium distance, and how steep the "
                "repulsive wall is."
            ),
            kind="success",
        )
    else:
        _done = sorted(_results.keys())
        _msg = "Fill in the function bodies above, then run those cells."
        if _done:
            _msg += f" Working so far: {', '.join(_done)}."
        mo.callout(mo.md(_msg), kind="warn")
    return (r,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 5 — read the curves with NumPy

    Use `np.min` and `np.argmin` to find the well depth and equilibrium
    distance of the Lennard-Jones potential from the discrete grid.
    Then find the approximate zero crossing — the first grid point
    where $V$ changes sign.

    Replace the three `...` below:

    - `V_min`: use `np.min(V_lj)` — the smallest value in the array.
    - `i_min`: use `np.argmin(V_lj)` — the *index* of that smallest value.
    - `i_zero`: use `np.argmax(V_lj[:-1] * V_lj[1:] < 0)` — this
      multiplies neighbouring values; the product is negative at a
      sign change, so `argmax` finds the first `True`.
    """)
    return


@app.cell
def _(epsilon, lennard_jones, mo, np, r, sigma):
    try:
        _V_lj = lennard_jones(r, epsilon, sigma)
        if _V_lj is ... or _V_lj is None:
            raise ValueError
        _V_lj = np.asarray(_V_lj, dtype=float)
    except Exception:
        mo.callout(
            mo.md("Finish Step 1 (Lennard-Jones) first — this step uses that function."),
            kind="warn",
        )
        mo.stop(True)

    V_lj = _V_lj

    # --- Fill in the three lines below ---
    V_min = ...          # np.min(V_lj)
    i_min = ...          # np.argmin(V_lj)
    i_zero = ...         # np.argmax(V_lj[:-1] * V_lj[1:] < 0)
    # -------------------------------------

    try:
        _v = float(V_min)
        _i = int(i_min)
        _iz = int(i_zero)
        _r_min = r[_i]
        _r_zero = r[_iz]
    except Exception:
        mo.callout(
            mo.md("Replace the three `...` above with the NumPy calls described in the instructions."),
            kind="warn",
        )
        mo.stop(True)

    mo.callout(
        mo.md(
            f"""
            **Lennard-Jones on a {len(r)}-point grid**

            - Well depth: $V_{{\\min}} = {_v:.6f}$ eV at $r = {_r_min:.3f}$ $\\mathrm{{\\AA}}$
            - Analytical equilibrium: $r_{{\\min}} = 2^{{1/6}}\\sigma = {2**(1/6) * 2.55:.3f}$ $\\mathrm{{\\AA}}$
            - Approximate zero crossing: $r \\approx {_r_zero:.3f}$ $\\mathrm{{\\AA}}$

            These are only as accurate as the grid spacing. Later in the
            course we will find these values with proper numerical methods.
            """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
