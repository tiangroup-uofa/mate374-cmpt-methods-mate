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
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("a + b:", a + b)
    print("2 * a:", 2 * a)
    return a, b


@app.cell
def _(a, b, np):
    print("a * b:", a * b)           # Multiply matching entries
    print("a @ b:", a @ b)           # Multiply, then add
    print("np.dot(a, b):", np.dot(a, b))
    return


@app.cell
def _(a, np):
    print("By hand:", np.sqrt(a[0]**2 + a[1]**2 + a[2]**2))
    print("From the dot product:", np.sqrt(a @ a))
    print("With NumPy:", np.linalg.norm(a))
    return


if __name__ == "__main__":
    app.run()
