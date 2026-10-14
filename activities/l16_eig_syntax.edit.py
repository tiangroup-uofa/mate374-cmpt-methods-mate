# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def matrix(np):
    H = np.array([[4.0, 6.0], [6.0, 4.0]])
    H
    return (H,)


@app.cell(hide_code=True)
def general_intro(mo):
    mo.md("""
    **General square matrix: np.linalg.eig**

    The eigenvectors are columns, in the same order as the eigenvalues.
    This example has real results. Check the imaginary parts before using
    the real part with another matrix.
    """)
    return


@app.cell
def general_case(H, mo, np):
    values, vectors = np.linalg.eig(H)
    imaginary_parts_vanish = (
        np.allclose(values.imag, 0) and np.allclose(vectors.imag, 0)
    )
    # Use the real part only when the imaginary parts are negligible.
    real_values = values.real if imaginary_parts_vanish else values
    real_vectors = vectors.real if imaginary_parts_vanish else vectors
    mo.vstack([
        mo.md(f"Imaginary parts vanish: **{imaginary_parts_vanish}**"),
        mo.md("Eigenvalues"), real_values,
        mo.md("Eigenvectors (columns)"), real_vectors,
    ])
    return


@app.cell(hide_code=True)
def symmetric_intro(mo):
    mo.md("""
    **Symmetric matrix: np.linalg.eigh**

    The eigenvalues are sorted from smallest to largest.
    Column i is the unit eigenvector for eigenvalue i.
    Try changing both off-diagonal entries together to keep the matrix symmetric.
    """)
    return


@app.cell
def symmetric_case(H, mo, np):
    symmetric_values, symmetric_vectors = np.linalg.eigh(H)
    mo.vstack([
        mo.md("Eigenvalues"), symmetric_values,
        mo.md("Eigenvectors (columns)"), symmetric_vectors,
        mo.md("Check: H @ vectors − vectors * values"),
        H @ symmetric_vectors - symmetric_vectors * symmetric_values,
    ])
    return


if __name__ == "__main__":
    app.run()
