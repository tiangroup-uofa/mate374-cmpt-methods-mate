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


@app.cell
def _(np):
    def random_chains(n_segments, n_chains, seed=374):
        """Ideal chains: each segment has length b = 1 and a random direction in 3D."""
        rng = np.random.default_rng(seed)
        steps = rng.normal(size=(n_chains, n_segments, 3))
        steps /= np.linalg.norm(steps, axis=2, keepdims=True)
        return np.cumsum(steps, axis=1)          # bead positions, shape (chains, N, 3)


    def rms_end_to_end(n_segments, n_chains):
        R = random_chains(n_segments, n_chains)[:, -1, :]
        return np.sqrt(np.mean(np.sum(R**2, axis=1)))

    return random_chains, rms_end_to_end


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(steps=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000], value=500, label="segments N", show_value=True)
    m_slider = mo.ui.slider(steps=[20, 50, 100, 200, 500], value=200, label="number of chains", show_value=True)
    return m_slider, n_slider


@app.cell(hide_code=True)
def _(m_slider, mo, n_slider, np, plt, random_chains, rms_end_to_end):
    N, M = n_slider.value, m_slider.value
    _chain = random_chains(N, 1)[0]
    _Ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    _R = np.array([rms_end_to_end(n, M) for n in _Ns])
    R_N = rms_end_to_end(N, M)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    _a.plot(_chain[:, 0], _chain[:, 1], lw=0.8, color="C0")
    _a.plot(*_chain[0, :2], "o", color="C2", label="start")
    _a.plot(*_chain[-1, :2], "s", color="C3", label="end")
    _a.set(aspect="equal", title=f"one chain, N = {N} (x-y projection)")
    _a.legend(fontsize=8)
    _b.loglog(_Ns, _R, "o", color="C0", label=f"random chains ({M} per N)")
    _b.loglog(_Ns, np.sqrt(_Ns), "--", color="0.4", label=r"$\sqrt{N}$")
    _b.loglog(_Ns, _Ns, ":", color="0.6", label="fully stretched, N")
    _b.loglog([N], [R_N], "*", ms=14, color="k")
    _b.set(xlabel="segments N", ylabel="rms end-to-end distance (units of b)")
    _b.legend(fontsize=8)
    mo.vstack([mo.hstack([n_slider, m_slider]),
               mo.md(f"N = {N}: rms end-to-end distance **{R_N:.1f} b**, compared with √N = {np.sqrt(N):.1f} b and a stretched length of {N} b."),
               _fig])
    return


if __name__ == "__main__":
    app.run()
