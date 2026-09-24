# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline

    return CubicSpline, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## Four quadratic pieces through five points

    Each piece has the form $\hat f_i(x)=a_i x^2+b_i x+c_i$.
    Write the endpoint and slope conditions on paper and compare them with
    the supplied matrix. The boundary condition is $\hat f_1''(8)=0$.
    Edit existing cells rather than defining the same names in new cells.

    **Your calculation:** evaluate at `x_new = 20.0` after checking 12.7.
    Which polynomial should supply that value?
    """)
    return


@app.cell(hide_code=False)
def data(np):
    x = np.array([8., 11., 15., 18., 22.])
    y = np.array([5., 9., 10., 8., 7.])
    return x, y


@app.cell(hide_code=False)
def system(np):
    # Columns: a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4
    A = np.array([
        [64, 8, 1,   0, 0, 0,     0, 0, 0,     0, 0, 0],
        [121, 11, 1, 0, 0, 0,     0, 0, 0,     0, 0, 0],
        [0, 0, 0,    121, 11, 1,  0, 0, 0,     0, 0, 0],
        [0, 0, 0,    225, 15, 1,  0, 0, 0,     0, 0, 0],
        [0, 0, 0,    0, 0, 0,     225, 15, 1,  0, 0, 0],
        [0, 0, 0,    0, 0, 0,     324, 18, 1,  0, 0, 0],
        [0, 0, 0,    0, 0, 0,     0, 0, 0,     324, 18, 1],
        [0, 0, 0,    0, 0, 0,     0, 0, 0,     484, 22, 1],
        [22, 1, 0,   -22, -1, 0,  0, 0, 0,     0, 0, 0],
        [0, 0, 0,    30, 1, 0,    -30, -1, 0,  0, 0, 0],
        [0, 0, 0,    0, 0, 0,     36, 1, 0,    -36, -1, 0],
        [2, 0, 0,    0, 0, 0,     0, 0, 0,     0, 0, 0],
    ], dtype=float)
    return (A,)


@app.cell(hide_code=False)
def right_hand_side(np, y):
    # Each interior data value appears at the end of one piece and start of the next.
    rhs = np.array([y[0], y[1], y[1], y[2], y[2], y[3], y[3], y[4], 0, 0, 0, 0])
    return (rhs,)


@app.cell(hide_code=False)
def solve(A, np, rhs):
    coefficients = np.linalg.solve(A, rhs)
    pieces = coefficients.reshape(4, 3)
    print("Rows are [a_i, b_i, c_i]:")
    print(pieces)
    print("Largest equation residual:", np.max(np.abs(A @ coefficients - rhs)))
    return coefficients, pieces


@app.cell(hide_code=False)
def evaluate(np, pieces):
    x_new = 12.7
    piece_index = 1  # The second piece uses row 1; valid for 11 <= x_new <= 15.
    estimate = np.polyval(pieces[piece_index], x_new)
    print(f"f_hat_{piece_index + 1}({x_new:g}) = {estimate:.8f}")
    return estimate, piece_index, x_new


@app.cell(hide_code=False)
def checks(np, pieces, x, y):
    endpoint_errors = np.array([
        np.polyval(pieces[i], x[i:i + 2]) - y[i:i + 2]
        for i in range(4)
    ])
    slope_jumps = np.array([
        (2 * pieces[i, 0] * x[i + 1] + pieces[i, 1])
        - (2 * pieces[i + 1, 0] * x[i + 1] + pieces[i + 1, 1])
        for i in range(3)
    ])
    print("Endpoint errors:", endpoint_errors)
    print("Left slope minus right slope:", slope_jumps)
    print("Left-end second derivative:", 2 * pieces[0, 0])
    return endpoint_errors, slope_jumps


@app.cell(hide_code=False)
def library_comparison(CubicSpline, x, x_new, y):
    cubic = CubicSpline(x, y, bc_type="natural", extrapolate=False)
    print(f"Natural cubic at {x_new:g}: {float(cubic(x_new)):.8f}")
    return (cubic,)


@app.cell(hide_code=True)
def plot(cubic, estimate, mo, np, piece_index, pieces, plt, x, x_new, y):
    def plot_spline():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i in range(4):
            grid = np.linspace(x[i], x[i + 1], 100)
            ax.plot(grid, np.polyval(pieces[i], grid), label=f"Quadratic piece {i + 1}")
        dense_x = np.linspace(x[0], x[-1], 400)
        ax.plot(dense_x, cubic(dense_x), "k--", alpha=0.6, label="Natural cubic")
        ax.scatter(x, y, color="black", zorder=4, label="Supplied data")
        if 0 <= piece_index < 4 and x[piece_index] <= x_new <= x[piece_index + 1]:
            ax.scatter([x_new], [estimate], marker="*", s=130, color="crimson", zorder=5)
        ax.set(xlabel="x", ylabel="y", title="Quadratic interpolation through five points")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    spline_figure = plot_spline()
    interval_message = (
        f"Selected piece covers [{x[piece_index]:g}, {x[piece_index + 1]:g}]."
        if 0 <= piece_index < 4 else "Select a piece_index from 0 to 3."
    )
    mo.vstack([mo.md(interval_message + " Check that it contains `x_new`."), spline_figure])
    return (spline_figure,)


@app.cell(hide_code=True)
def follow_up(mo):
    mo.md(r"""
    **Your calculation:** set `x_new = 20.0` and `piece_index = 3`.
    Then evaluate at 15 using piece indices 1 and 2. Do the values agree?

    The natural cubic matches both first and second derivatives at interior
    points and sets the second derivative to zero at both ends. These conditions
    differ from our quadratic construction, so the estimates between data
    points can differ.
    """)
    return


if __name__ == "__main__":
    app.run()
