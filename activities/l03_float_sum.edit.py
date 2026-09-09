# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    print(0.1 + 0.2)
    print(0.1 + 0.2 == 0.3)
    return


if __name__ == "__main__":
    app.run()
