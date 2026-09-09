# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    print(bin(13))
    print(int("1101", 2))
    return


if __name__ == "__main__":
    app.run()
