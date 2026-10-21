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
def _():
    def f(x):
        return x**3


    def df_exact(x):
        return 3 * x**2

    return df_exact, f


@app.cell
def _(mo):
    h_slider = mo.ui.slider(steps=[1.0, 0.5, 0.25, 0.1, 0.05, 0.01], value=1.0, label="h", show_value=True)
    h_slider
    return (h_slider,)


@app.cell
def _(df_exact, f, h_slider):
    x = 3.0
    h = h_slider.value

    forward = (f(x + h) - f(x)) / h
    backward = (f(x) - f(x - h)) / h
    central = (f(x + h) - f(x - h)) / (2 * h)

    exact = df_exact(x)
    for name, value in [("forward", forward), ("backward", backward), ("central", central)]:
        print(f"{name:9s} {value:12.6f}   error = {value - exact:+.3e}")
    print(f"{'exact':9s} {exact:12.6f}")
    return


if __name__ == "__main__":
    app.run()
