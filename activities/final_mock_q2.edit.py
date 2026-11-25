# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock final · Q2")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


if __name__ == "__main__":
    app.run()
