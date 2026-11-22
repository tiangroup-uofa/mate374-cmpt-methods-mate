# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    t_tc = mo.ui.slider(steps=[0.5, 0.7, 0.9, 1.1, 1.5, 3.0], value=0.7, label="T / Tc", show_value=True)
    ag = mo.ui.slider(0.1, 0.5, step=0.05, value=0.5, label="Ag fraction", show_value=True)
    sweeps = mo.ui.slider(steps=[20, 50, 100, 200, 400], value=100, label="sweeps", show_value=True)
    seed = mo.ui.number(start=0, stop=99_999_999, step=1, value=374, label="seed")
    run = mo.ui.run_button(label="Run")
    mo.vstack([mo.hstack([t_tc, ag], justify="start", wrap=True),
               mo.hstack([sweeps, seed, run], justify="start", wrap=True)])
    return ag, run, seed, sweeps, t_tc


@app.cell
def _(np):
    TC = 1.0 / np.log(1.0 + np.sqrt(2.0))      # k Tc / epsilon for the square lattice

    def unlike_around(s, r, c, L):
        """Number of unlike bonds between site (r, c) and its four neighbours."""
        v = s[r][c]
        return ((s[(r + 1) % L][c] != v) + (s[(r - 1) % L][c] != v)
                + (s[r][(c + 1) % L] != v) + (s[r][(c - 1) % L] != v))

    def boltzmann_swaps(L, ag_fraction, kT, n_sweeps, seed):
        """L27's lattice loop, with the swap accepted by the Boltzmann rule."""
        rng = np.random.default_rng(seed)
        flat = [1 if k < round(ag_fraction * L * L) else 0 for k in range(L * L)]
        rng.shuffle(flat)
        s = [flat[r * L:(r + 1) * L] for r in range(L)]   # s[r][c] = 1 for Ag, 0 for Cu
        start = [row[:] for row in s]
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        unlike_history = []
        for sweep in range(n_sweeps):
            for attempt in range(2 * L * L):              # one attempt per bond
                r, c = int(rng.integers(L)), int(rng.integers(L))
                dr, dc = steps[int(rng.integers(4))]
                r2, c2 = (r + dr) % L, (c + dc) % L
                if s[r][c] == s[r2][c2]:
                    continue                              # swapping like atoms changes nothing
                before = unlike_around(s, r, c, L) + unlike_around(s, r2, c2, L)
                s[r][c], s[r2][c2] = s[r2][c2], s[r][c]
                after = unlike_around(s, r, c, L) + unlike_around(s, r2, c2, L)
                dE = after - before                       # in units of epsilon
                if dE > 0 and rng.random() >= np.exp(-dE / kT):
                    s[r][c], s[r2][c2] = s[r2][c2], s[r][c]   # rejected: swap back
            unlike_history.append(sum(unlike_around(s, r, c, L) for r in range(L) for c in range(L)) / (4 * L * L))
        return np.array(start), np.array(s), np.array(unlike_history)

    return TC, boltzmann_swaps


@app.cell(hide_code=True)
def _(TC, ag, boltzmann_swaps, mo, run, seed, sweeps, t_tc, time):
    mo.stop(not run.value, mo.md("Choose the inputs and press **Run**."))
    _t0 = time.perf_counter()
    start, final, unlike = boltzmann_swaps(32, ag.value, TC * t_tc.value, sweeps.value, seed.value)
    elapsed = time.perf_counter() - _t0
    return elapsed, final, start, unlike


@app.cell(hide_code=True)
def _(ag, elapsed, final, mo, np, plt, start, sweeps, unlike):
    _fig, _ax = plt.subplots(1, 3, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for _a, _g, _t in [(_ax[0], start, "start"), (_ax[1], final, f"after {sweeps.value} sweeps")]:
        _a.imshow(_g, cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
        _a.set_title(_t, fontsize=10); _a.set_xticks([]); _a.set_yticks([])
    _c = ag.value
    _ax[2].plot(np.arange(1, len(unlike) + 1), unlike, color="#1f77b4", lw=1.4, label="simulation")
    _ax[2].axhline(2 * _c * (1 - _c), color="0.5", ls="--", lw=1, label="random mixture")
    _ax[2].set(xlabel="sweeps", ylabel="unlike-bond fraction", ylim=(0, 0.6))
    _ax[2].legend(frameon=False, fontsize=9)
    _ax[2].spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _attempts = 2 * 32 * 32 * sweeps.value
    mo.vstack([_fig, mo.md(f"{_attempts:,} attempts in {elapsed:.1f} s: "
                           f"**{_attempts / elapsed:,.0f}** attempts per second in pure Python.")])
    return


if __name__ == "__main__":
    app.run()
