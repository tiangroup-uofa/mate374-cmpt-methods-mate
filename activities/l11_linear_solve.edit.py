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
    # Two moving atoms connected to fixed atom 0 by springs of 5 eV/Å².
    K = np.array([[10.0, -5.0],
                  [-5.0,  5.0]])       # eV/Å²
    # Double or reverse the end load and predict the displacements.
    F = np.array([0.0, 0.5])            # eV/Å
    return F, K


@app.cell
def _(F, K, np):
    K_inverse = np.linalg.inv(K)         # Å²/eV
    u_inverse = K_inverse @ F            # Å
    print("Inverse:\n", K_inverse)
    print("K_inverse @ K:\n", K_inverse @ K)
    print("Displacements from the inverse (Å):", u_inverse)
    print("Entrywise reciprocals K**-1:\n", K**-1)
    return


@app.cell
def _(F, K, np):
    u = np.linalg.solve(K, F)            # Å
    residual = K @ u - F                 # eV/Å
    print("Displacements from solve (Å):", u)
    print("Force-balance residual (eV/Å):", residual)
    print("Maximum absolute residual (eV/Å):", np.max(np.abs(residual)))
    return


if __name__ == "__main__":
    app.run()
