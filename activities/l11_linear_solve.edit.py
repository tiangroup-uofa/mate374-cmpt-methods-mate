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
    # --- 3 atoms (2 moving), k = 5 eV/Å², F_ext = 10 eV/Å ---
    K2 = np.array([[10.0, -5.0],
                   [-5.0,  5.0]])
    F2 = np.array([0.0, 10.0])
    K2_inv = np.linalg.inv(K2)
    u2 = K2_inv @ F2
    print("K (2×2):\n", K2)
    print("K inverse:\n", K2_inv)
    print("Displacements (Å):", u2)
    print("Check K @ u - F:", K2 @ u2 - F2)
    return


@app.cell
def _(np):
    # --- 4 atoms (3 moving), k = 5 eV/Å², F_ext = 10 eV/Å ---
    K3 = np.array([[10.0, -5.0,  0.0],
                   [-5.0, 10.0, -5.0],
                   [ 0.0, -5.0,  5.0]])
    F3 = np.array([0.0, 0.0, 10.0])
    K3_inv = np.linalg.inv(K3)
    u3 = K3_inv @ F3
    print("K (3×3):\n", K3)
    print("K inverse:\n", K3_inv)
    print("Displacements (Å):", u3)
    print("Check K @ u - F:", K3 @ u3 - F3)
    return


if __name__ == "__main__":
    app.run()
