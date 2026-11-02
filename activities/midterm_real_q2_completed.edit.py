# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "scipy"]
# ///
import marimo

app = marimo.App(width="medium")

@app.cell
def _():
    import numpy as np
    from scipy.optimize import root_scalar
    return np, root_scalar

@app.cell
def _(np):
    Tc, Pc, omega, R = 304.1282, 73.7703, 0.22394, 0.0831446
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    b = 0.07780 * R * Tc / Pc

    def a(T):
        return 0.45724 * R**2 * Tc**2 / Pc * (
            1 + kappa * (1 - np.sqrt(T / Tc))
        )**2

    def F(V, T, P):
        return R*T/(V-b) - a(T)/(V**2 + 2*b*V - b**2) - P

    return F, a, b

@app.cell
def _(F):
    print("Check residual (bar):", F(1.0, 400.0, 1.0))
    return

@app.cell
def _(F, root_scalar):
    brackets = [(0.045, 0.065), (0.10, 0.14), (0.35, 0.45)]
    for bracket in brackets:
        solution = root_scalar(
            F, args=(280.0, 40.0), bracket=bracket,
            method="bisect", xtol=1e-12,
        )
        print("Converged:", solution.converged,
              "V (L/mol):", solution.root,
              "Residual (bar):", F(solution.root, 280.0, 40.0))
    return

if __name__ == "__main__":
    app.run()
