# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    scale = mo.ui.radio(
        options=["linear", "log–log"],
        value="log–log",
        label="Axis scale",
        inline=True,
    )
    scale
    return (scale,)


@app.cell(hide_code=True)
def _(np):
    rng = np.random.default_rng(374)
    maximum_throws = 100_000

    centre_distance = 0.5 * rng.random(maximum_throws)
    angle = 0.5 * np.pi * rng.random(maximum_throws)
    cumulative_crossings = np.cumsum(
        centre_distance <= 0.5 * np.sin(angle)
    )

    sampled_throws = np.unique(
        np.geomspace(10, maximum_throws, 90).astype(int)
    )
    sampled_crossings = cumulative_crossings[sampled_throws - 1]
    estimates = 2.0 * sampled_throws / sampled_crossings
    observed_error = np.abs(estimates - np.pi)

    analytical_throws = np.geomspace(10, 1_000_000_000, 300)
    analytical_sigma = 2.3735 / np.sqrt(analytical_throws)
    return analytical_sigma, analytical_throws, observed_error, sampled_throws


@app.cell(hide_code=True)
def _(
    analytical_sigma,
    analytical_throws,
    observed_error,
    plt,
    sampled_throws,
    scale,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        sampled_throws,
        observed_error,
        color="#275D38",
        linewidth=1.4,
        marker="o",
        markersize=3,
        label="one simulated sequence",
    )
    ax.plot(
        analytical_throws,
        analytical_sigma,
        color="#D97706",
        linewidth=2,
        label=r"large-sample scale, $2.37/\sqrt{N}$",
    )
    ax.axhline(
        5.0e-5,
        color="#7C3AED",
        linestyle="--",
        linewidth=1.3,
        label=r"five-significant-digit target, $5\times10^{-5}$",
    )

    if scale.value == "log–log":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1.0e-5, 2.0)
    else:
        ax.set_xlim(0, 1_000_000_000)
        ax.set_ylim(0, 0.85)

    ax.set_xlabel("Number of throws, $N$")
    ax.set_ylabel(r"Absolute error or expected scale")
    ax.set_title("Buffon estimate: observed error and analytical scaling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
