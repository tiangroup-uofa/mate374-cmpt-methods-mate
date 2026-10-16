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
    H = np.array([[2.0, -1.0, 0.0],
                  [-1.0, 2.0, -1.0],
                  [0.0, -1.0, 2.0]])
    values, vectors = np.linalg.eigh(H)

    # Intended: check H v = lambda v for the first eigenpair
    v = vectors[0]
    print("eigenvalue:", values[0])
    print("H v - lambda v:", H @ v - values[0] * v)
    return


if __name__ == "__main__":
    app.run()
