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
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(np):
    def mc_estimate(d, M, rng):
        """Monte Carlo estimate of the integral of x1^2 + ... + xd^2 over the unit cube."""
        x = rng.random((M, d))
        f = np.sum(x**2, axis=1)
        return f.mean(), f.std(ddof=1)

    return (mc_estimate,)


@app.cell(hide_code=True)
def _(mo):
    d_slider = mo.ui.slider(
        steps=[1, 2, 3, 5, 10, 20, 30], value=10, label="dimension d", show_value=True
    )
    m_slider = mo.ui.slider(
        steps=[10, 30, 100, 300, 1000, 3000, 10000],
        value=1000,
        label="samples M",
        show_value=True,
    )
    return d_slider, m_slider


@app.cell(hide_code=True)
def _(d_slider, m_slider, mc_estimate, mo, np, plt):
    d, M = d_slider.value, m_slider.value
    exact = d / 3
    rng = np.random.default_rng(374)
    repeats = np.array([mc_estimate(d, M, rng)[0] for _ in range(300)])
    rel = (repeats - exact) / exact
    predicted = 0.894 / np.sqrt(d * M)

    Ms = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    rms = []
    for m in Ms:
        est = np.array([mc_estimate(d, m, rng)[0] for _ in range(100)])
        rms.append(np.sqrt(np.mean(((est - exact) / exact) ** 2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained")
    ax1.hist(rel, bins=30, color="#cde5ec", edgecolor="#1f77b4")
    for s in (-predicted, predicted):
        ax1.axvline(s, color="#b5473a", ls="--", lw=1.5)
    ax1.set(
        xlabel="relative error of one estimate",
        ylabel="count of 300 repeated runs",
        title=f"d = {d}, M = {M}",
    )
    ax2.loglog(Ms, rms, "o", color="#1f77b4", label="repeated runs (rms)")
    ax2.loglog(Ms, 0.894 / np.sqrt(d * Ms), "--", color="#b5473a", label=r"$0.894/\sqrt{dM}$")
    ax2.loglog([M], [predicted], "*", ms=13, color="#231f20")
    ax2.set(xlabel="samples M", ylabel="relative standard error")
    ax2.legend(frameon=False)

    mo.vstack(
        [
            mo.hstack([d_slider, m_slider], justify="start"),
            mo.md(
                f"Predicted relative standard error **{predicted:.4f}**. "
                f"A grid with 10 points per coordinate needs **10<sup>{d}</sup>** evaluations."
            ),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
