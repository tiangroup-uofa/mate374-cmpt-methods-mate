# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    return ListedColormap, math, mo, np, plt, time


@app.cell
def _(math, np):
    def random_alloy(L, rng):
        """Half A (0) and half B (1) on an L x L periodic lattice."""
        s = np.array([0] * (L * L // 2) + [1] * (L * L // 2), dtype=np.int8)
        rng.shuffle(s)
        return s.reshape(L, L)

    def unlike_fraction(s):
        n_ab = np.sum(s != np.roll(s, 1, 0)) + np.sum(s != np.roll(s, 1, 1))
        return n_ab / (2 * s.size)

    def serial_sweep(s, kT, rng):
        """2 L^2 attempts, one at a time: pick a site and a neighbour, try to swap."""
        L = s.shape[0]
        for _ in range(2 * L * L):
            r, c = int(rng.integers(L)), int(rng.integers(L))
            r2, c2 = (r, (c + 1) % L) if rng.random() < 0.5 else ((r + 1) % L, c)
            p, q = s[r, c], s[r2, c2]
            if p == q:
                continue
            d_e = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                y, x = (r + dr) % L, (c + dc) % L
                if (y, x) != (r2, c2):
                    d_e += int(s[y, x] != q) - int(s[y, x] != p)
                y, x = (r2 + dr) % L, (c2 + dc) % L
                if (y, x) != (r, c):
                    d_e += int(s[y, x] != p) - int(s[y, x] != q)
            if d_e <= 0 or rng.random() < math.exp(-d_e / kT):
                s[r, c], s[r2, c2] = q, p
        return s

    def pair_set_sweep(s, kT, rng):
        """2 L^2 attempts in 16 passes. Each pass updates one set of
        non-touching pairs at once, using whole-array operations. L must be a
        multiple of 4."""
        rows, cols = np.indices(s.shape)
        for axis in (1, 0):
            t = s if axis == 1 else s.T  # view: pairs run along the second index
            for a in range(4):
                for b in range(2):
                    q = np.roll(t, -1, 1)  # right partner of each site
                    p_nb = (np.roll(t, 1, 1), np.roll(t, 1, 0), np.roll(t, -1, 0))
                    q_nb = (np.roll(t, -2, 1), np.roll(q, 1, 0), np.roll(q, -1, 0))
                    before = sum(n != t for n in p_nb) + sum(n != q for n in q_nb)
                    after = sum(n != q for n in p_nb) + sum(n != t for n in q_nb)
                    d_e = after - before
                    in_set = (cols % 4 == a) & (rows % 2 == b) & (t != q)
                    draw = rng.random(t.shape)
                    accept = in_set & ((d_e <= 0) | (draw < np.exp(-d_e / kT)))
                    new = np.where(accept, q, t)
                    t[...] = np.where(np.roll(accept, 1, 1), np.roll(t, 1, 1), new)
        return s

    return pair_set_sweep, random_alloy, serial_sweep, unlike_fraction


@app.cell(hide_code=True)
def _(mo):
    size_choice = mo.ui.dropdown(
        options={"32 × 32": 32, "64 × 64": 64, "128 × 128": 128},
        value="64 × 64",
        label="lattice",
    )
    kT_slider = mo.ui.slider(
        steps=[1.0, 1.25, 1.6, 2.0, 4.0], value=1.6, label="kT / ε", show_value=True
    )
    return kT_slider, size_choice


@app.cell(hide_code=True)
def _(
    ListedColormap,
    kT_slider,
    mo,
    np,
    pair_set_sweep,
    plt,
    random_alloy,
    serial_sweep,
    size_choice,
    time,
    unlike_fraction,
):
    L, kT = size_choice.value, kT_slider.value

    s_serial = random_alloy(L, np.random.default_rng(374))
    t0 = time.perf_counter()
    for _ in range(3):
        serial_sweep(s_serial, kT, np.random.default_rng(1))
    t_serial = (time.perf_counter() - t0) / 3

    rng = np.random.default_rng(374)
    s_par = random_alloy(L, rng)
    record = []
    t0 = time.perf_counter()
    for _ in range(300):
        pair_set_sweep(s_par, kT, rng)
        record.append(unlike_fraction(s_par))
    t_par = (time.perf_counter() - t0) / 300

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.6, 3.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]}
    )
    ax1.imshow(s_par, cmap=ListedColormap(["#d0d2d6", "#1f77b4"]), interpolation="nearest")
    ax1.set(xticks=[], yticks=[], title="after 300 parallel sweeps")
    ax2.plot(np.arange(1, 301), record, color="#231f20")
    ax2.axhline(0.5, color="#8a8f98", ls="--", label="random mixture")
    ax2.set(xlabel="sweep", ylabel="unlike-bond fraction", ylim=(0, 0.6))
    ax2.legend(frameon=False)

    mo.vstack(
        [
            mo.hstack([size_choice, kT_slider], justify="start"),
            mo.md(
                f"One sweep = {2 * L * L:,} attempts. Serial loop **{t_serial * 1e3:.1f} ms**, "
                f"pair sets **{t_par * 1e3:.2f} ms**, speedup **{t_serial / t_par:.0f}×**. "
                f"Mean of the last 150 sweeps **{np.mean(record[150:]):.3f}**."
            ),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
