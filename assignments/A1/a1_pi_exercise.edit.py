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
    # A1 Q3 · Estimate π

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
    # You can increast the largest N if you want
    N_values = [1, 2, 4, 8, 16, 32, 64]
    parameters = {}  # For example: {"seed": 374} for a random method.
    return N_values, parameters


@app.cell(hide_code=True)
def _(mo):
    show_known = mo.ui.switch(value=False, label="Compare with the four L02 methods")
    show_known
    return (show_known,)


@app.cell(hide_code=True)
def _(np):
    def known_method_errors():
        import math

        # Fixed, modest sample counts keep the optional comparison browser-friendly.
        counts = np.array([10, 100, 1000, 10000, 100000])
        rng = np.random.default_rng(374)
        distance = 0.5 * rng.random(counts[-1])
        angle = 0.5 * np.pi * rng.random(counts[-1])
        crossings = np.cumsum(distance <= 0.5 * np.sin(angle))[counts - 1]
        buffon = np.divide(2.0 * counts, crossings,
                          out=np.full(len(counts), np.nan), where=crossings > 0)
        terms = np.arange(1, counts[-1] + 1, dtype=float)
        basel = np.sqrt(6 * np.cumsum(1 / terms**2)[counts - 1])
        side = 1.0
        sides = 6
        polygon_counts, polygon = [], []
        for _ in range(25):
            polygon_counts.append(sides)
            polygon.append(sides * side / 2)
            side = math.sqrt(2 - math.sqrt(4 - side * side))
            sides *= 2
        total = 0.0
        ramanujan = []
        for k in range(4):
            total += math.factorial(4*k) * (1103 + 26390*k) / (math.factorial(k)**4 * 396**(4*k))
            ramanujan.append(9801 / (2 * math.sqrt(2) * total))
        return [
            ("Buffon (throws; seed 374)", counts, np.abs(buffon - np.pi)),
            ("Basel (terms)", counts, np.abs(basel - np.pi)),
            ("Polygon (sides)", np.array(polygon_counts), np.abs(np.array(polygon) - np.pi)),
            ("Ramanujan (terms)", np.arange(1, 5), np.abs(np.array(ramanujan) - np.pi)),
        ]
    return (known_method_errors,)


@app.cell(hide_code=True)
def _(N_values, estimate_pi, known_method_errors, mo, np, parameters, plt, show_known):
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
        ax.loglog(np.asarray(N_values)[positive], errors[positive], "o-", lw=2, label="Your method")
        if show_known.value:
            for name, counts, reference_errors in known_method_errors():
                visible = (reference_errors > 0) & np.isfinite(reference_errors)
                ax.loglog(counts[visible], reference_errors[visible], ".--", alpha=.8, label=name)
        ax.legend(fontsize=8)
        ax.set_xlabel("Effort parameter N")
        ax.set_ylabel("Absolute error against np.pi")
        ax.set_title("A1 Q3 · Convergence comparison")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        note = mo.md(
            "Zero and non-finite errors are not drawn on the logarithmic plot; "
            "inspect the table too. A zero here means agreement with the "
            "binary64 reference after conversion to binary64. "
            "It does not establish additional correct digits. "
            "The optional L02 curves use their own sample counts; N means different work "
            "in each method, so this is not a runtime comparison. All curves here use np.pi "
            "as the reference, including Ramanujan."
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
