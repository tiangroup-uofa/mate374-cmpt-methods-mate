# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    def solve_by_inversion(K, F):
        """Return displacements satisfying K @ u = F."""
        return np.linalg.inv(K) @ F

    return (solve_by_inversion,)


@app.cell
def _(np, solve_by_inversion):
    # --- 3 atoms (2 moving), k = 5 eV/Å², F_ext = 10 eV/Å ---
    K2 = np.array([[10.0, -5.0],
                   [-5.0,  5.0]])
    F2 = np.array([0.0, 10.0])
    u2 = solve_by_inversion(K2, F2)
    print("Displacements (Å):", u2)
    print("Check K @ u - F:", K2 @ u2 - F2)
    return


@app.cell
def _(np, solve_by_inversion):
    # --- 4 atoms (3 moving), k = 5 eV/Å², F_ext = 10 eV/Å ---
    K3 = np.array([[10.0, -5.0,  0.0],
                   [-5.0, 10.0, -5.0],
                   [ 0.0, -5.0,  5.0]])
    F3 = np.array([0.0, 0.0, 10.0])
    u3 = solve_by_inversion(K3, F3)
    print("Displacements (Å):", u3)
    print("Check K @ u - F:", K3 @ u3 - F3)
    return


if __name__ == "__main__":
    app.run()
