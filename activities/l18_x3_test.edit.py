# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    h_choice = mo.ui.dropdown(
        options=["1e-1", "1e-2", "1e-3", "1e-5", "1e-8", "1e-10", "1e-13", "1e-15"],
        value="1e-1",
        label="h",
    )
    return (h_choice,)


@app.cell
def _(h_choice, mo):
    def f(x):
        return x**3


    x = 3.0
    h = float(h_choice.value)
    central = (f(x + h) - f(x - h)) / (2 * h)

    mo.hstack(
        [
            h_choice,
            mo.md(
                f"""
    | | value |
    |---|---|
    | central difference | `{central!r}` |
    | error | `{central - 27:.3e}` |
    | predicted $h^2$ | `{h**2:.3e}` |
    """
            ),
        ],
        justify="start",
        align="center",
        gap=3,
    )
    return


if __name__ == "__main__":
    app.run()
