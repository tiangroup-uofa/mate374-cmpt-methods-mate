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
    h = 1.0
    x = np.arange(0.0, 6.0, h)       # x_0, x_1, ..., x_(n-1)
    fx = x**3                         # Sampled values f(x_i)
    n = len(fx)
    print("x  =", x)
    print("fx =", fx)
    return fx, h, n, x


@app.cell
def _(fx, h, n, np):
    # Way 1: visit one index at a time
    df_loop = np.empty_like(fx)
    for i in range(1, n - 1):        # Interior points: central difference
        df_loop[i] = (fx[i + 1] - fx[i - 1]) / (2 * h)
    # End points: three-point forward and backward differences
    df_loop[0] = (-3 * fx[0] + 4 * fx[1] - fx[2]) / (2 * h)
    df_loop[n - 1] = (3 * fx[n - 1] - 4 * fx[n - 2] + fx[n - 3]) / (2 * h)
    print("loop   :", df_loop)
    return (df_loop,)


@app.cell
def _(df_loop, fx, h, np, x):
    # Way 2: one array operation for all interior points
    df_slice = np.empty_like(fx)
    df_slice[1:-1] = (fx[2:] - fx[:-2]) / (2 * h)
    df_slice[0] = (-3 * fx[0] + 4 * fx[1] - fx[2]) / (2 * h)
    df_slice[-1] = (3 * fx[-1] - 4 * fx[-2] + fx[-3]) / (2 * h)
    print("slicing:", df_slice)
    print("exact  :", 3 * x**2)
    print("same as loop:", np.allclose(df_loop, df_slice))
    return


@app.cell
def _(fx, h, np, x):
    # Second derivative at the interior points
    d2f = (fx[2:] - 2 * fx[1:-1] + fx[:-2]) / h**2
    print("f'' interior:", d2f)
    print("exact 6x    :", 6 * x[1:-1])
    return


if __name__ == "__main__":
    app.run()
