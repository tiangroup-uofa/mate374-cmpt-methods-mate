# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib>=3.9",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    return Line2D, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Buffon's needle: estimating π with random throws

    Imagine a floor marked by parallel lines. Drop many identical needles
    onto it and count how many cross a line. The crossing fraction contains
    enough geometric information to estimate π.

    Choose your inputs, then press **Simulate**. Changing a control does not
    run a new experiment until you press the button again.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    seed_number = mo.ui.number(
        start=0,
        stop=99_999_999,
        step=1,
        value=374,
        label="Random seed (enter your student ID; at most 8 digits)",
    )
    length_ratio = mo.ui.slider(
        start=0.10,
        stop=1.00,
        step=0.05,
        value=0.75,
        include_input=True,
        label="Needle length / line spacing, L/D",
        show_value=True,
    )
    number_of_needles = mo.ui.slider(
        start=10,
        stop=1_000_000,
        step=10,
        value=5_000,
        include_input=True,
        label="Number of virtual needles, N",
        show_value=True,
    )
    simulate = mo.ui.run_button(label="Simulate")

    mo.vstack(
        [seed_number, length_ratio, number_of_needles, simulate],
        align="start",
        gap=1,
    )
    return length_ratio, number_of_needles, seed_number, simulate


@app.cell(hide_code=True)
def _(np):
    def simulate_buffon(seed, length_to_spacing, sample_count):
        """Generate virtual needles and return crossings and π estimates."""
        rng = np.random.default_rng(seed)

        # Set the line spacing D = 1. The needle length is then L/D.
        center_x = rng.uniform(0.0, 1.0, sample_count)
        center_y = rng.uniform(0.0, 1.0, sample_count)
        angle = rng.uniform(0.0, np.pi, sample_count)

        distance_to_line = np.minimum(center_y, 1.0 - center_y)
        vertical_half_length = 0.5 * length_to_spacing * np.abs(np.sin(angle))
        crossed = distance_to_line <= vertical_half_length

        throw_number = np.arange(1, sample_count + 1)
        cumulative_crossings = np.cumsum(crossed)
        running_estimate = np.full(sample_count, np.nan)
        has_crossing = cumulative_crossings > 0
        running_estimate[has_crossing] = (
            2.0
            * length_to_spacing
            * throw_number[has_crossing]
            / cumulative_crossings[has_crossing]
        )

        return {
            "center_x": center_x,
            "center_y": center_y,
            "angle": angle,
            "crossed": crossed,
            "throw_number": throw_number,
            "cumulative_crossings": cumulative_crossings,
            "running_estimate": running_estimate,
        }

    return (simulate_buffon,)


@app.cell(hide_code=True)
def _(
    Line2D,
    length_ratio,
    mo,
    np,
    number_of_needles,
    plt,
    seed_number,
    simulate,
    simulate_buffon,
):
    if not simulate.value:
        output = mo.callout(
            mo.md(
                "Choose a seed, $L/D$, and $N$, then press **Simulate**. "
                "The default seed `374` is only a placeholder."
            ),
            kind="info",
        )
    else:
        seed = int(seed_number.value)
        ratio = float(length_ratio.value)
        sample_count = int(number_of_needles.value)
        data = simulate_buffon(seed, ratio, sample_count)

        crossings = int(data["cumulative_crossings"][-1])
        if crossings > 0:
            pi_estimate = 2.0 * ratio * sample_count / crossings
            absolute_error = abs(pi_estimate - np.pi)
            relative_percent_error = 100.0 * absolute_error / np.pi
        else:
            pi_estimate = np.nan
            absolute_error = np.nan
            relative_percent_error = np.nan

        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        geometry_axis, trend_axis = axes

        display_count = min(sample_count, 500)
        display_indices = np.linspace(
            0,
            sample_count - 1,
            display_count,
            dtype=int,
        )
        center_x = data["center_x"][display_indices]
        center_y = data["center_y"][display_indices]
        angle = data["angle"][display_indices]
        displayed_crossings = data["crossed"][display_indices]

        half_dx = 0.5 * ratio * np.cos(angle)
        half_dy = 0.5 * ratio * np.sin(angle)
        for crossed, x_value, y_value, dx_value, dy_value in zip(
            displayed_crossings,
            center_x,
            center_y,
            half_dx,
            half_dy,
        ):
            geometry_axis.plot(
                [x_value - dx_value, x_value + dx_value],
                [y_value - dy_value, y_value + dy_value],
                color="#D55E00" if crossed else "#0072B2",
                linewidth=1.5 if crossed else 1.0,
                alpha=0.90 if crossed else 0.55,
            )

        geometry_axis.axhline(0.0, color="black", linewidth=2.0)
        geometry_axis.axhline(1.0, color="black", linewidth=2.0)
        geometry_axis.set(
            xlim=(-0.08, 1.08),
            ylim=(-0.08, 1.08),
            aspect="equal",
            xticks=[],
            ylabel="Normalized line spacing, D = 1",
            title=f"A sample of {display_count} virtual needles",
        )
        geometry_axis.legend(
            handles=[
                Line2D([0], [0], color="#0072B2", lw=2, label="No crossing"),
                Line2D([0], [0], color="#D55E00", lw=2, label="Crossed a line"),
            ],
            loc="upper right",
            frameon=False,
        )

        valid_trend = (
            (data["throw_number"] >= 10)
            & np.isfinite(data["running_estimate"])
        )
        trend_axis.plot(
            data["throw_number"][valid_trend],
            data["running_estimate"][valid_trend],
            color="#0072B2",
            linewidth=1.4,
            label=r"Running estimate, $\hat{\pi}$",
        )
        trend_axis.axhline(
            np.pi,
            color="#D55E00",
            linestyle="--",
            linewidth=1.6,
            label=r"Reference, $\pi$",
        )
        if crossings > 0:
            trend_axis.scatter(
                [sample_count],
                [pi_estimate],
                color="#0072B2",
                edgecolor="white",
                linewidth=0.8,
                s=55,
                zorder=3,
            )
        trend_axis.set_xscale("log")
        trend_axis.set(
            xlabel="Number of virtual needles, N",
            ylabel=r"Estimate of $\pi$",
            title="How the estimate changes during the run",
        )
        trend_axis.grid(alpha=0.25)
        trend_axis.legend(frameon=False)

        figure.suptitle(
            rf"Buffon's needle simulation: $L/D={ratio:.2f}$, $N={sample_count:,}$",
            fontsize=13,
        )
        figure.tight_layout()

        if crossings > 0:
            result_lines = fr"""
            - crossings: **{crossings:,} out of {sample_count:,} needles**
            - estimate: **$\hat{{\pi}}={pi_estimate:.8f}$**
            - absolute error: **$|\hat{{\pi}}-\pi|={absolute_error:.3e}$**
            - relative percentage error: **{relative_percent_error:.4f}%**
            """
        else:
            result_lines = f"""
            - crossings: **0 out of {sample_count:,} needles**
            - estimate: **undefined because the crossing count is zero**

            Increase $N$ or $L/D$, then press **Simulate** again.
            """

        results = mo.md(
            f"""
            ## Result

            {result_lines}

            Your seed was accepted but is not repeated in the output. The
            calculation runs locally in this browser notebook. Keep the same
            seed and settings if you want to reproduce this exact run.
            """
        )
        output = mo.vstack([figure, results], gap=1)
    output
    return


if __name__ == "__main__":
    app.run()
