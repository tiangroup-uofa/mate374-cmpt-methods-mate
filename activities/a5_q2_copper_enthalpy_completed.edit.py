# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A5 Q2 · Enthalpy of copper from its heat capacity

    $$
    \Delta H=\int_{300\ \mathrm K}^{1300\ \mathrm K}C_p(T)\,\mathrm dT
    $$

    Complete **Step 1** and **Step 2** in the calculation cell. The same cell
    contains the NumPy and SciPy calls for comparison.
    """)
    return


@app.cell(hide_code=True)
def supplied_table(np):
    # Heat capacity of solid copper, J/(mol K), every 100 K from 300 K to 1300 K
    T = np.arange(300.0, 1301.0, 100.0)
    Cp = np.array([24.49, 25.29, 25.99, 26.55, 27.02, 27.47,
                   27.97, 28.62, 29.48, 30.64, 32.18])
    h = T[1] - T[0]
    return Cp, T, h


@app.cell(hide_code=True)
def data_table(Cp, T, mo):
    mo.md("| $T$ (K) | " + " | ".join(f"{t:.0f}" for t in T) + " |\n"
          + "|---" * (len(T) + 1) + "|\n"
          + "| $C_p$ (J/(mol K)) | " + " | ".join(f"{c:.2f}" for c in Cp) + " |")
    return


@app.cell(hide_code=True)
def steps_text(mo):
    mo.md(r"""
    ### **Step 1** · Composite trapezoidal rule

    $$
    I\approx\frac h2\left(y_0+2y_1+2y_2+\cdots+2y_{N-1}+y_N\right)
    $$

    ### **Step 2** · Composite Simpson's 1/3 rule

    $$
    I\approx\frac h3\left(y_0+4y_1+2y_2+4y_3+\cdots+4y_{N-1}+y_N\right)
    $$

    Use NumPy slices, as in
    [Lecture 20](https://tiangroup-uofa.github.io/mate374-cmpt-methods-mate/units/04/L20/index.html):
    `Cp[1:-1]` holds every interior value, `Cp[1:-1:2]` the values with weight 4,
    and `Cp[2:-1:2]` the values with weight 2.
    """)
    return


@app.cell
def steps(Cp, T, h, np):
    from numpy import trapezoid  # Gold standard: trapezoid
    from scipy.integrate import simpson  # General Simpson method

    # Step 1: replace np.nan with the composite trapezoidal rule
    dH_trap = h / 2 * (np.sum(Cp) + np.sum(Cp[1:-1]))
    dH_trap_reference = trapezoid(Cp, T)

    # Step 2: replace np.nan with the composite Simpson's 1/3 rule
    dH_simp = h / 3 * (np.sum(Cp) + np.sum(Cp[1:-1]) + 2 * np.sum(Cp[1:-1:2]))
    dH_simp_reference = simpson(Cp, x=T)
    # Equivalent: h / 3 * (Cp[0] + Cp[-1] + 4 * np.sum(Cp[1:-1:2]) + 2 * np.sum(Cp[2:-1:2]))

    return dH_simp, dH_simp_reference, dH_trap, dH_trap_reference


@app.cell(hide_code=True)
def check(dH_simp, dH_simp_reference, dH_trap, dH_trap_reference, mo, np):
    _rows = []
    for _name, _value, _ref in [
        ("Trapezoidal", dH_trap, dH_trap_reference),
        ("Simpson's 1/3", dH_simp, dH_simp_reference),
    ]:
        _status = "Pass" if np.isclose(_value, _ref) else "Not yet"
        _rows.append(f"| {_name} | {_value:.1f} | {_ref:.1f} | {_status} |")
    mo.md("| Rule | Your result (J/mol) | Library result (J/mol) | Check |\n"
          "|---|---:|---:|---|\n" + "\n".join(_rows))
    return


if __name__ == "__main__":
    app.run()
