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


@app.cell(hide_code=True)
def _(mo):
    p_slider = mo.ui.slider(
        steps=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01], value=0.1,
        label="probability of a hop per vibration, p", show_value=True,
    )
    atoms = mo.ui.slider(
        steps=[10, 100, 1000, 10000], value=1000, label="number of atoms", show_value=True,
    )
    seed = mo.ui.number(start=0, stop=99_999_999, step=1, value=374, label="seed")
    mo.hstack([p_slider, atoms, seed], justify="start", wrap=True)
    return atoms, p_slider, seed


@app.cell
def _(np):
    def vibrations_until_hop(p, rng):
        """Count vibrations until one of them ends in a hop."""
        count = 0
        while True:
            count += 1
            if rng.random() < p:  # an event that happens with probability p
                return count

    return (vibrations_until_hop,)


@app.cell
def _(atoms, np, p_slider, seed, vibrations_until_hop):
    rng = np.random.default_rng(seed.value)
    waits = np.array([vibrations_until_hop(p_slider.value, rng) for _ in range(atoms.value)])
    return (waits,)


@app.cell(hide_code=True)
def _(atoms, mo, np, p_slider, plt, waits):
    _p = p_slider.value
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=110)
    _k = np.arange(1, waits.max() + 1)
    ax.hist(waits, bins=np.arange(0.5, waits.max() + 1.5), density=True,
            color="#9ecae1", edgecolor="#3182bd", lw=0.4, label="simulated atoms")
    ax.plot(_k, (1 - _p) ** (_k - 1) * _p, color="#d62728", lw=1.5,
            label=r"exact: $(1-p)^{k-1}\,p$")
    ax.set(xlabel="vibrations until the hop, k", ylabel="fraction of atoms",
           xlim=(0, min(waits.max(), 8 / _p) + 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    mo.vstack([
        fig,
        mo.md(f"Average wait over {atoms.value:,} atoms: **{waits.mean():.1f}** vibrations. "
              f"Exact average 1/*p* = **{1 / _p:.0f}**."),
    ])
    return


if __name__ == "__main__":
    app.run()
