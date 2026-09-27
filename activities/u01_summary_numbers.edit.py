# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # bin() and int() used to convert decimal and binary integers
    print(bin(13))
    print(int("1101", 2))
    return


@app.cell
def _(np):
    # numpy allows explicit definition of float32 and float64
    import numpy as np
    x_single = np.float32(0.1)
    x_double = np.float64(0.1)
    print(f"float32: {x_single:.20f}")
    print(f"float64: {x_double:.20f}")
    print("Default array dtype:", np.array([0.1, 0.2]).dtype)
    return


if __name__ == "__main__":
    app.run()
