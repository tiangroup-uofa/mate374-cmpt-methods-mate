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
    size = mo.ui.dropdown(options=["10", "20", "40"], value="20", label="lattice L × L, L =")
    ag = mo.ui.slider(0.1, 0.5, step=0.05, value=0.2, label="Ag fraction", show_value=True)
    p_hop = mo.ui.slider(0.1, 1.0, step=0.1, value=0.5, label="hop probability p", show_value=True)
    sweeps = mo.ui.slider(steps=[10, 50, 100, 200, 500], value=100, label="sweeps", show_value=True)
    seed = mo.ui.number(start=0, stop=99_999_999, step=1, value=374, label="seed")
    run = mo.ui.run_button(label="Run")
    mo.vstack([
        mo.hstack([size, ag, p_hop], justify="start", wrap=True),
        mo.hstack([sweeps, seed, run], justify="start", wrap=True),
    ])
    return ag, p_hop, run, seed, size, sweeps


@app.cell
def _(np):
    def lattice_hops(L, ag_fraction, p, n_sweeps, seed):
        """Monte Carlo hops on an L x L periodic square lattice, one attempt at a time.

        site[i] is the label of the atom on site i. x[a], y[a] are the unwrapped
        coordinates of atom a, so they keep counting when an atom crosses the edge.
        """
        rng = np.random.default_rng(seed)
        N = L * L
        is_ag = [k < round(ag_fraction * N) for k in range(N)]
        rng.shuffle(is_ag)                        # random starting arrangement
        site = list(range(N))                     # atom a starts on site a
        x = [i % L for i in range(N)]
        y = [i // L for i in range(N)]
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        start = (x[:], y[:])
        msd = [0.0]
        for sweep in range(n_sweeps):
            for attempt in range(N):              # one sweep = N attempts
                i = int(rng.integers(N))          # pick a random site
                dx, dy = steps[int(rng.integers(4))]
                j = (i % L + dx) % L + ((i // L + dy) % L) * L   # its neighbour
                if rng.random() < p:              # the hop happens with probability p
                    a, b = site[i], site[j]
                    site[i], site[j] = b, a       # the two atoms exchange sites
                    x[a] += dx; y[a] += dy
                    x[b] -= dx; y[b] -= dy
            msd.append(sum((x[a] - start[0][a]) ** 2 + (y[a] - start[1][a]) ** 2
                           for a in range(N)) / N)
        grid = np.array([is_ag[site[i]] for i in range(N)]).reshape(L, L)
        return np.array(msd), grid, np.array(is_ag).reshape(L, L)

    return (lattice_hops,)


@app.cell(hide_code=True)
def _(ag, lattice_hops, mo, p_hop, run, seed, size, sweeps, time):
    mo.stop(not run.value, mo.md("Choose the inputs and press **Run**."))
    _t0 = time.perf_counter()
    msd, final_grid, start_grid = lattice_hops(int(size.value), ag.value, p_hop.value,
                                               sweeps.value, seed.value)
    elapsed = time.perf_counter() - _t0
    return elapsed, final_grid, msd, start_grid


@app.cell(hide_code=True)
def _(elapsed, final_grid, mo, msd, np, p_hop, plt, size, start_grid, sweeps):
    _L = int(size.value)
    _t = np.arange(len(msd))
    _slope = np.polyfit(_t[len(_t) // 5:], msd[len(_t) // 5:], 1)[0]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=110,
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for _ax, _g, _title in [(axes[0], start_grid, "start"), (axes[1], final_grid, f"after {sweeps.value} sweeps")]:
        _ax.imshow(_g, cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
        _ax.set_title(_title, fontsize=10); _ax.set_xticks([]); _ax.set_yticks([])
    axes[2].plot(_t, msd, color="#1f77b4", lw=1.4, label="simulation")
    axes[2].plot(_t, 2 * p_hop.value * _t, "--", color="#d62728", lw=1.2, label=r"$2p$ per sweep")
    axes[2].set(xlabel="sweeps", ylabel=r"$\langle R^2\rangle$ / $a^2$")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    _attempts = sweeps.value * _L * _L
    mo.vstack([
        fig,
        mo.md(
            f"Fitted slope: **{_slope:.3f}** \\(a^2\\) per sweep, so "
            f"\\(D = \\text{{slope}}/4 =\\) **{_slope / 4:.3f}** \\(a^2\\) per sweep.  \n"
            f"{_attempts:,} attempts in {elapsed:.2f} s: "
            f"**{_attempts / elapsed:,.0f}** attempts per second in pure Python."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
