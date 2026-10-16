# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy"]
# ///

# This notebook opens with a deliberate error for students to fix.
# mate374: build-execute = false

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import scipy as sp
    return (np, sp)


@app.cell
def _(np):
    D0 = 2.3e-5      # m^2/s, carbon in FCC iron
    Q = 148e3        # J/mol
    R = 8.314        # J/(mol K)

    def diffusivity(T):
        return D0 * np.exp(-Q / (R * T))

    def diffusion_length(T, t):
        return np.sqrt(diffusivity(T) * t)

    def residual(T, t, L_target):
        return diffusion_length(T, t) - L_target
    return (residual,)


@app.cell
def _(residual, sp):
    # Intended: the temperature at which L = 0.5 mm after 1 h
    t = 3600.0          # s
    L_target = 0.5e-3   # m

    solution = sp.optimize.root_scalar(
        residual(1200.0, t, L_target), bracket=[1000.0, 1600.0], method="bisect"
    )
    solution.root
    return


if __name__ == "__main__":
    app.run()
