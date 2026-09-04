# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib>=3.9",
# ]
# ///

# The comparison intentionally performs long pure-Python loops after a click.
# mate374: build-execute = false

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import math
    import random
    import time

    return math, mo, plt, random, time


@app.cell(hide_code=True)
def _(mo):
    run_comparison = mo.ui.run_button(label="Run the four-method comparison")
    mo.vstack(
        [
            run_comparison,
            mo.md(
                "The implementations use ordinary Python. The largest loops "
                "contain $10^7$ steps, so the browser may remain busy for a while."
            ),
        ]
    )
    return (run_comparison,)


@app.cell(hide_code=True)
def _(math, random):
    checkpoints = [10**power for power in range(1, 8)]

    def buffon_errors():
        rng = random.Random(374)
        crossings = 0
        errors = []
        checkpoint_index = 0
        for throw in range(1, checkpoints[-1] + 1):
            centre_distance = 0.5 * rng.random()
            angle = 0.5 * math.pi * rng.random()
            if centre_distance <= 0.5 * math.sin(angle):
                crossings += 1
            if throw == checkpoints[checkpoint_index]:
                estimate = 2.0 * throw / crossings
                errors.append(abs(estimate - math.pi))
                checkpoint_index += 1
                if checkpoint_index == len(checkpoints):
                    break
        return checkpoints, errors

    def basel_errors():
        total = 0.0
        errors = []
        checkpoint_index = 0
        for k in range(1, checkpoints[-1] + 1):
            total += 1.0 / (k * k)
            if k == checkpoints[checkpoint_index]:
                estimate = math.sqrt(6.0 * total)
                errors.append(abs(estimate - math.pi))
                checkpoint_index += 1
                if checkpoint_index == len(checkpoints):
                    break
        return checkpoints, errors

    def polygon_errors():
        sides = 6
        side_length = 1.0
        side_counts = [sides]
        errors = [abs(sides * side_length / 2.0 - math.pi)]
        while 2 * sides <= 10_000_000:
            side_length = math.sqrt(
                2.0 - math.sqrt(4.0 - side_length * side_length)
            )
            sides *= 2
            side_counts.append(sides)
            errors.append(abs(sides * side_length / 2.0 - math.pi))
        return side_counts, errors

    def ramanujan_errors():
        term_counts = [1, 2]
        errors = []
        for number_of_terms in term_counts:
            total = 0.0
            for k in range(number_of_terms):
                numerator = math.factorial(4 * k) * (1103 + 26390 * k)
                denominator = math.factorial(k) ** 4 * 396 ** (4 * k)
                total += numerator / denominator
            inverse_pi = 2.0 * math.sqrt(2.0) * total / 9801.0
            errors.append(abs(1.0 / inverse_pi - math.pi))
        return term_counts, errors

    return basel_errors, buffon_errors, polygon_errors, ramanujan_errors


@app.cell(hide_code=True)
def _(
    basel_errors,
    buffon_errors,
    mo,
    plt,
    polygon_errors,
    ramanujan_errors,
    run_comparison,
    time,
):
    if not run_comparison.value:
        comparison_output = mo.callout(
            mo.md("Press the button to calculate and compare the four methods."),
            kind="info",
        )
    else:
        start = time.perf_counter()
        buffon_N, buffon_error = buffon_errors()
        basel_N, basel_error = basel_errors()
        polygon_N, polygon_error = polygon_errors()
        ramanujan_N, ramanujan_error = ramanujan_errors()
        elapsed = time.perf_counter() - start

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.loglog(buffon_N, buffon_error, "o-", label="Buffon")
        ax.loglog(basel_N, basel_error, "o-", label="Basel")
        ax.loglog(polygon_N, polygon_error, "o-", label="polygon")
        ax.loglog(ramanujan_N, ramanujan_error, "o-", label="Ramanujan")
        ax.set_xlabel("Effort parameter, $N$")
        ax.set_ylabel(r"Absolute error, $|\pi_N-\pi|$")
        ax.set_title("Four routes to the same number")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()

        comparison_output = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"The pure-Python comparison finished in "
                        f"**{elapsed:.2f} s** on this browser."
                    ),
                    kind="success",
                ),
                fig,
            ]
        )

    comparison_output
    return


if __name__ == "__main__":
    app.run()
