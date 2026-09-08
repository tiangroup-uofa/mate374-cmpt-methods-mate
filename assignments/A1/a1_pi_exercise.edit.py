# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A1: choose a method for computing π

    Find a numerical method different from the L02 examples using an AI
    tool, a search engine, or the literature. Explain its equation and what its effort parameter $N$
    counts. Record your source in your submission.

    Complete `estimate_pi` below. The supplied code compares its output
    with `np.pi`, the same binary64 reference used in the L02 comparison
    notebook. This reference has finite precision: matching it is not
    proof that your answer equals mathematical π exactly.
    """)
    return


@app.cell
def _():
    def estimate_pi(N: int, **parameters):
        # Replace ... with your method and return one numerical estimate.
        # You may add imports and helper functions in this cell.
        return ...

    return (estimate_pi,)


@app.cell
def _():
    # Adapt these values to your method. Start with inexpensive calculations.
    N_values = [1, 2, 4, 8, 16, 32, 64]
    parameters = {}  # For example: {"seed": 374} for a random method.
    return N_values, parameters


@app.cell(hide_code=True)
def _(N_values, estimate_pi, mo, np, parameters, plt):
    def compare_estimates():
        estimates = [estimate_pi(N, **parameters) for N in N_values]
        if any(value is ... for value in estimates):
            return mo.callout(
                mo.md("Complete `estimate_pi`, then run that cell. The error table and plot will appear here."),
                kind="warn",
            )

        # This supplied comparison uses binary64, not arbitrary precision.
        values = np.asarray(estimates, dtype=np.float64)
        errors = np.abs(values - np.pi)
        rows = [
            {"N": N, "Estimate": float(value), "Absolute error against np.pi": float(error),
             "Relative error against np.pi": float(error / abs(np.pi))}
            for N, value, error in zip(N_values, values, errors)
        ]
        positive = (errors > 0) & np.isfinite(errors)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(np.asarray(N_values)[positive], errors[positive], "o-")
        ax.set_xlabel("Effort parameter N")
        ax.set_ylabel("Absolute error against np.pi")
        ax.set_title("Convergence of your π method")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        note = mo.md(
            "Zero and non-finite errors are not drawn on the logarithmic plot; "
            "inspect the table too. A zero here means agreement with the "
            "binary64 reference after conversion to binary64. "
            "It does not establish additional correct digits."
        )
        return mo.vstack([mo.ui.table(rows, selection=None), fig, note])

    compare_estimates()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpret your result

    - Change `N_values` to reveal how your method converges. What does $N$ count?
    - Compare absolute and relative error as $N$ increases. Relative error
      is reported as a fraction, not a percentage.
    - Does the error decrease? Explain the trend you observe, including
      plateaus, fluctuations, or failures.

    If you cannot find or implement a suitable method, ask the instructor
    for the fallback method. You can still carry out the convergence study.
    """)
    return


if __name__ == "__main__":
    app.run()
