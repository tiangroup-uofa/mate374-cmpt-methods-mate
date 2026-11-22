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
    def metropolis_two_state(kT, attempts, seed=374):
        """States 0 and 1 with energies 0 and 1 (units of epsilon).

        Each attempt proposes a switch or a stay with equal probability.
        Every attempt records the current state, including rejections.
        """
        rng = np.random.default_rng(seed)
        state = 0
        records = np.empty(attempts, dtype=int)
        accepted_switch = np.zeros(attempts, dtype=bool)
        for k in range(attempts):
            if rng.random() < 0.5:  # propose a switch
                dE = 1.0 if state == 0 else -1.0
                if dE <= 0 or rng.random() < np.exp(-dE / kT):
                    state = 1 - state
                    accepted_switch[k] = True
            records[k] = state
        return records, accepted_switch

    return (metropolis_two_state,)


@app.cell(hide_code=True)
def _(mo):
    kT_slider = mo.ui.slider(
        steps=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0, label="kT / ε", show_value=True
    )
    count_mode = mo.ui.dropdown(
        options=["every attempt", "accepted switches only"],
        value="every attempt",
        label="records counted",
    )
    return count_mode, kT_slider


@app.cell(hide_code=True)
def _(count_mode, kT_slider, metropolis_two_state, mo, np, plt):
    kT = kT_slider.value
    records, accepted = metropolis_two_state(kT, 20000)
    exact = np.exp(-1 / kT) / (1 + np.exp(-1 / kT))

    if count_mode.value == "every attempt":
        counted = records
    else:
        counted = records[accepted]
    running = np.cumsum(counted) / np.arange(1, counted.size + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    ax1.step(np.arange(80), records[:80], where="post", color="#231f20")
    ax1.set(yticks=[0, 1], xlabel="attempt", ylabel="recorded state", title="first 80 records")
    ax2.semilogx(np.arange(1, counted.size + 1), running, color="#1f77b4", label="sampled fraction")
    ax2.axhline(exact, color="#b5473a", ls="--", label="Boltzmann value")
    ax2.set(xlabel="number of counted records", ylabel="fraction in state 1", ylim=(0, 0.8))
    ax2.legend(frameon=False)

    mo.vstack(
        [
            mo.hstack([kT_slider, count_mode], justify="start"),
            mo.md(f"Sampled **{running[-1]:.3f}**, Boltzmann value **{exact:.3f}**."),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
