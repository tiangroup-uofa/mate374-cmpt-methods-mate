# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy"]
# ///
import marimo

app = marimo.App(width="medium")

@app.cell
def _():
    import numpy as np
    return (np,)

@app.cell
def _(np):
    A = np.array([[4, 1, 3, 2],
                  [8, 2, 4, 2],
                  [0, 7, 2, 9],
                  [0, 2, 9, 0]], dtype=float)
    b = np.array([21, 34, 29, 31], dtype=float)
    return A, b

@app.cell
def _(A, b, np):
    x = np.linalg.solve(A, b)
    print("Amounts:", x)
    print("Calculated intensities:", A @ x)
    print("Check:", np.allclose(A @ x, b))
    return (x,)

@app.cell
def _(A, np, x):
    b_true = np.array([22, 34, 29, 31], dtype=float)
    x_true = np.linalg.solve(A, b_true)
    e_rel = np.abs(x - x_true) / x_true
    print("Corrected amounts:", x_true)
    print("Absolute relative errors:", e_rel)
    print("Percentage errors:", 100 * e_rel)
    print("Sensor percentage error:", 100 / 22)
    print("Condition number:", np.linalg.cond(A))
    return

if __name__ == "__main__":
    app.run()
