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

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Root finding by bisection

    Solve $x^2-c=0$ on $[0,4]$ and compare with the exact root $\sqrt{c}$.
    """)
    return


@app.cell
def _(mo):
    target = mo.ui.slider(
        start=1.0,
        stop=15.0,
        step=0.5,
        value=2.0,
        label="Target, c",
        show_value=True,
    )
    target
    return (target,)


@app.cell
def _(target):
    left = 0.0
    right = 4.0
    iterations = 30

    for _ in range(iterations):
        midpoint = 0.5 * (left + right)
        if midpoint * midpoint < target.value:
            left = midpoint
        else:
            right = midpoint

    root = 0.5 * (left + right)
    residual = abs(root * root - target.value)
    return iterations, residual, root


@app.cell
def _(iterations, mo, residual, root, target):
    mo.md(
        f"""
        **Bisection result**

        - Root: $x = {root:.10f}$
        - Residual: $|x^2-c| = {residual:.2e}$
        - Iterations: {iterations}
        - Exact comparison: $\\sqrt{{c}} = {target.value ** 0.5:.10f}$
        """
    )
    return


if __name__ == "__main__":
    app.run()
