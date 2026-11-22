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
    U = mo.ui.slider(0.0, 4.0, step=0.25, value=2.0, label="repulsion U (eV)", show_value=True)
    d_eps = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="ε_Ag − ε_Cu (eV)", show_value=True)
    alpha = mo.ui.slider(steps=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], value=0.3, label="mixing α", show_value=True)
    mo.hstack([U, d_eps, alpha], justify="start", wrap=True)
    return U, alpha, d_eps


@app.cell
def _(np):
    def ring_matrix(eps_site, t=1.0):
        """Tight-binding matrix of a periodic ring with on-site energies eps_site."""
        N = len(eps_site)
        H = np.diag(eps_site).astype(float)
        for i in range(N):
            H[i, (i + 1) % N] = H[(i + 1) % N, i] = -t
        return H

    def occupations(H):
        """Fill the lowest N/2 levels with two electrons; return electrons on each site."""
        N = H.shape[0]
        E, C = np.linalg.eigh(H)
        return 2 * (C[:, : N // 2] ** 2).sum(axis=1)

    def self_consistent(eps0, U, alpha, max_iter=200, tol=1e-8):
        """The on-site energy depends on the electrons on that site: eps_i = eps0_i + U (n_i - 1).
        Guess n, build H, solve, get a new n, mix, repeat."""
        n = np.ones(len(eps0))                       # guess: one electron on every atom
        history = []
        for k in range(max_iter):
            n_new = occupations(ring_matrix(eps0 + U * (n - 1)))
            change = np.abs(n_new - n).max()
            history.append(change)
            if change < tol:
                break
            n = (1 - alpha) * n + alpha * n_new      # linear mixing of old and new density
        return n, np.array(history)

    return (self_consistent,)


@app.cell
def _(U, alpha, d_eps, np, self_consistent):
    N = 12
    eps0 = np.array([0.0, d_eps.value] * (N // 2))   # alternating Cu, Ag
    n_final, history = self_consistent(eps0, U.value, alpha.value)
    n_frozen, _ = self_consistent(eps0, 0.0, 1.0)     # same ring without the repulsion
    return N, history, n_final, n_frozen


@app.cell(hide_code=True)
def _(N, history, mo, n_final, n_frozen, np, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
    _ax[0].semilogy(np.arange(1, len(history) + 1), np.maximum(history, 1e-12), "o-", ms=3, color="#c26a2b")
    _ax[0].axhline(1e-8, color="0.5", ls=":", lw=1)
    _ax[0].set(xlabel="iteration", ylabel="largest change in n", title="self-consistency loop")
    _ax[0].spines[["top", "right"]].set_visible(False)
    _x = np.arange(N)
    _ax[1].bar(_x - 0.2, n_frozen, 0.4, color="0.75", label="U = 0 (no feedback)")
    _ax[1].bar(_x + 0.2, n_final, 0.4, color=["#c26a2b", "#5f6f80"] * (N // 2), label="self-consistent")
    _ax[1].axhline(1, color="k", lw=0.6)
    _ax[1].set_xticks(_x, ["Cu", "Ag"] * (N // 2), fontsize=8)
    _ax[1].set(ylabel="electrons on the site", ylim=(0, 2), title="electron density on each atom")
    _ax[1].legend(frameon=False, fontsize=8)
    _ax[1].spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _ok = history[-1] < 1e-8
    mo.vstack([_fig, mo.md(
        (f"Converged in **{len(history)}** iterations." if _ok else
         f"**Not converged** after {len(history)} iterations: try a smaller mixing α.")
        + f" Charge on Cu: **{n_final[0]:.3f}**, on Ag: **{n_final[1]:.3f}** electrons.")])
    return


if __name__ == "__main__":
    app.run()
