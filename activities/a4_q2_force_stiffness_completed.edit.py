# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A4 Q2 · Completed reference notebook

    Force and stiffness of a Cu–Cu pair from a table of energies.

    - force: $F(r)=-\mathrm dE/\mathrm dr$ (eV/Å)
    - stiffness: $k(r)=\mathrm d^2E/\mathrm dr^2$ (eV/Å²)
    """)
    return


@app.cell
def supplied_table(np):
    # Cu–Cu pair energy E (eV) at separations r (Å), with the exact force and
    # stiffness at the same separations.
    r_data = np.round(2.4 + 0.1 * np.arange(10), 2)
    E_data = np.array([-0.07516, -0.20055, -0.27790, -0.32095, -0.33988,
                       -0.34220, -0.33340, -0.31746, -0.29726, -0.27484])
    F_exact = np.array([1.5510, 0.9873, 0.5824, 0.2954, 0.0956,
                        -0.0402, -0.1293, -0.1847, -0.2159, -0.2302])
    k_exact = np.array([6.600, 4.765, 3.400, 2.390, 1.645,
                        1.100, 0.704, 0.419, 0.217, 0.077])
    return E_data, F_exact, k_exact, r_data


@app.cell(hide_code=True)
def data_table(E_data, F_exact, k_exact, mo, r_data):
    mo.md("| $r$ (Å) | $E$ (eV) | exact $F$ (eV/Å) | exact $k$ (eV/Å²) |\n|---:|---:|---:|---:|\n"
          + "\n".join(f"| {r:.2f} | {E:.5f} | {F:.4f} | {k:.3f} |"
                      for r, E, F, k in zip(r_data, E_data, F_exact, k_exact)))
    return


@app.cell(hide_code=True)
def step_one(mo):
    mo.md(r"""
    ### **Step 1** · Second derivative

    The function below already calculates the first derivative `yd` at every point
    with $O(h^2)$ formulas. Complete the second derivative `ydd` in the same way:

    - interior points: $\dfrac{y_{i-1}-2y_i+y_{i+1}}{h^2}$
    - first point: $\dfrac{2y_0-5y_1+4y_2-y_3}{h^2}$
    - last point: $\dfrac{2y_{n-1}-5y_{n-2}+4y_{n-3}-y_{n-4}}{h^2}$
    """)
    return


@app.cell
def step_one_derivatives(np):
    def first_second_derivative(x, y):
        h = x[1] - x[0]
        yd = np.empty(len(y))
        ydd = np.empty(len(y))

        # First derivative (supplied)
        yd[1:-1] = (y[2:] - y[:-2]) / (2 * h)
        yd[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h)
        yd[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * h)

        # Step 1: second derivative
        ydd[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / h**2
        ydd[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / h**2
        ydd[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / h**2
        return yd, ydd

    return (first_second_derivative,)


@app.cell(hide_code=True)
def function_check(first_second_derivative, mo, np):
    # Every O(h^2) second-derivative formula is exact for a cubic polynomial.
    _x = np.arange(6.0)
    _yd, _ydd = first_second_derivative(_x, _x**3)
    _passed = np.allclose(_ydd, 6 * _x)
    mo.callout(mo.md("**Check passed:** your second derivative is correct." if _passed
                     else "**Check not passed yet:** compare your `ydd` lines with the formulas above."),
               kind="success" if _passed else "warn")
    return


@app.cell
def apply_to_table(E_data, first_second_derivative, r_data):
    _dE, k_fd = first_second_derivative(r_data, E_data)
    F_fd = -_dE
    return F_fd, k_fd


@app.cell(hide_code=True)
def results_table(F_exact, F_fd, k_exact, k_fd, mo, r_data):
    mo.md("| $r$ (Å) | $F$ (eV/Å) | exact $F$ | $F$ error | $k$ (eV/Å²) | exact $k$ | $k$ error |\n"
          "|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(
              f"| {r:.2f} | {F:.4f} | {Fx:.4f} | {F - Fx:+.4f} | {k:.3f} | {kx:.3f} | {k - kx:+.3f} |"
              for r, F, Fx, k, kx in zip(r_data, F_fd, F_exact, k_fd, k_exact)))
    return


@app.cell(hide_code=True)
def plot(E_data, F_exact, F_fd, k_exact, k_fd, plt, r_data):
    fig, axes = plt.subplots(3, 1, figsize=(6, 7.5), sharex=True)
    axes[0].plot(r_data, E_data, "o-", ms=4, lw=0.8, color="0.3")
    for _ax, _fd, _ex in [(axes[1], F_fd, F_exact), (axes[2], k_fd, k_exact)]:
        _ax.plot(r_data, _ex, "s", ms=7, mfc="none", color="0.3", label="exact")
        _ax.plot(r_data, _fd, "o", ms=4, color="tab:red", label="finite difference")
        _ax.axhline(0, color="0.8", lw=0.6)
    axes[0].set_ylabel("$E$ (eV)")
    axes[1].set_ylabel(r"$F=-\mathrm{d}E/\mathrm{d}r$ (eV/Å)")
    axes[2].set_ylabel(r"$k=\mathrm{d}^2E/\mathrm{d}r^2$ (eV/Å²)")
    axes[1].legend(frameon=False)
    axes[-1].set_xlabel("$r$ (Å)")
    fig.tight_layout()
    fig
    return (fig,)


if __name__ == "__main__":
    app.run()
