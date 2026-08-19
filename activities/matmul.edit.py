# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.16",
#     "numpy>=2.0",
# ]
#
# [tool.marimo.display]
# code_editor_font_size = 16
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    ### Matrix multiplication

    Choose a matrix size and inspect the deterministic product $A B$.
    """)
    return


@app.cell
def _(mo):
    matrix_size = mo.ui.slider(
        start=2,
        stop=6,
        value=3,
        label="Matrix size, n",
        show_value=True,
    )
    matrix_size
    return (matrix_size,)


@app.cell
def _(matrix_size, mo, np):
    n = matrix_size.value
    matrix_a = np.arange(1, n * n + 1).reshape(n, n)
    matrix_b = np.flipud(np.eye(n, dtype=int))
    product = matrix_a @ matrix_b

    mo.md(
        f"""
        **Result for two {n} × {n} matrices**

        $A B =$ `{product.tolist()}`

        Trust check: multiplying by the reversed identity matrix reverses the
        columns of $A$: **{np.array_equal(product, matrix_a[:, ::-1])}**.
        """
    )
    return


if __name__ == "__main__":
    app.run()
