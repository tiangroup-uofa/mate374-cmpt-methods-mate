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
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = x**2
    h = x[1] - x[0]
    return h, x, y


@app.cell
def _(h, np, x, y):
    # Equally spaced data, with at least three points.
    # Write the interior slice calculation yourself, then compare.
    derivative = np.empty_like(y, dtype=float)
    derivative[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    derivative[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h)
    derivative[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * h)

    print("Interior positions:", x[1:-1])
    print("Interior derivatives:", derivative[1:-1])
    print("Full derivative array:", derivative)
    print("NumPy check:", np.gradient(y, h, edge_order=2))
    print("Exact derivative of x²:", 2 * x)
    return


if __name__ == "__main__":
    app.run()
