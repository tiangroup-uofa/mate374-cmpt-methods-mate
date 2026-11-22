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
    n_atoms = mo.ui.slider(steps=[8, 12, 20, 40], value=20, label="atoms in the ring, N", show_value=True)
    d_eps = mo.ui.slider(-1.0, 1.0, step=0.1, value=0.0, label="ε_Ag − ε_Cu (eV)", show_value=True)
    t_cucu = mo.ui.slider(0.5, 1.5, step=0.05, value=1.2, label="t_CuCu (eV)", show_value=True)
    t_agag = mo.ui.slider(0.5, 1.5, step=0.05, value=0.9, label="t_AgAg (eV)", show_value=True)
    t_cuag = mo.ui.slider(0.5, 1.5, step=0.05, value=0.8, label="t_CuAg (eV)", show_value=True)
    mo.vstack([
        mo.hstack([n_atoms, d_eps], justify="start", wrap=True),
        mo.hstack([t_cucu, t_agag, t_cuag], justify="start", wrap=True),
    ])
    return d_eps, n_atoms, t_agag, t_cuag, t_cucu


@app.cell
def _(np):
    def hamiltonian(types, eps, hop):
        """Tight-binding matrix of a periodic ring. types[i] is 'Cu' or 'Ag'."""
        N = len(types)
        H = np.zeros((N, N))
        for i in range(N):
            j = (i + 1) % N                       # right neighbour; the last atom bonds to the first
            H[i, i] = eps[types[i]]               # on-site energy
            t = hop[frozenset((types[i], types[j]))]
            H[i, j] = H[j, i] = -t                # hopping between neighbours
        return H

    def band_energy(types, eps, hop):
        """Fill the lowest N/2 levels with two electrons each (one s electron per atom)."""
        levels = np.linalg.eigvalsh(hamiltonian(types, eps, hop))
        return 2 * levels[: len(types) // 2].sum(), levels

    return (band_energy,)


@app.cell
def _(band_energy, d_eps, n_atoms, t_agag, t_cuag, t_cucu):
    N = n_atoms.value
    eps = {"Cu": 0.0, "Ag": d_eps.value}
    hop = {frozenset(["Cu"]): t_cucu.value, frozenset(["Ag"]): t_agag.value,
           frozenset(["Cu", "Ag"]): t_cuag.value}
    arrangements = {
        "alternating": ["Cu", "Ag"] * (N // 2),
        "two blocks": ["Cu"] * (N // 2) + ["Ag"] * (N // 2),
    }
    E_cu, levels_cu = band_energy(["Cu"] * N, eps, hop)
    E_ag, levels_ag = band_energy(["Ag"] * N, eps, hop)
    results = {}
    for _name, _types in arrangements.items():
        _E, _levels = band_energy(_types, eps, hop)
        _dH = _E / N - 0.5 * (E_cu + E_ag) / N          # mixing enthalpy per atom
        results[_name] = (_dH, _levels)
    return levels_ag, levels_cu, results


@app.cell(hide_code=True)
def _(levels_ag, levels_cu, mo, n_atoms, plt, results):
    _N = n_atoms.value
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    _sets = [("pure Cu", levels_cu, "#c26a2b"), ("pure Ag", levels_ag, "#5f6f80"),
             ("alternating", results["alternating"][1], "#1f77b4"),
             ("two blocks", results["two blocks"][1], "#2ca02c")]
    for _k, (_label, _lv, _col) in enumerate(_sets):
        for _i, _e in enumerate(_lv):
            _filled = _i < _N // 2
            _ax1.plot([_k - 0.35, _k + 0.35], [_e, _e], color=_col, lw=1.6 if _filled else 0.8,
                     alpha=1.0 if _filled else 0.35)
    _ax1.set_xticks(range(4), [_x[0] for _x in _sets], fontsize=9)
    _ax1.set_ylabel("electron energy level (eV)")
    _ax1.set_title("eigenvalues: filled (solid) and empty (faint)", fontsize=10)
    _ax1.spines[["top", "right"]].set_visible(False)
    _names = ["alternating", "two blocks"]
    _vals = [1000 * results[_n][0] for _n in _names]
    _ax2.bar(_names, _vals, color=["#1f77b4", "#2ca02c"])
    _ax2.axhline(0, color="k", lw=0.7)
    _ax2.set_ylabel("ΔH_mix (meV/atom)")
    _ax2.set_title("positive: separate · negative: mix", fontsize=10)
    _ax2.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"Alternating Cu–Ag: **{_vals[0]:+.0f} meV/atom**. "
              f"Two blocks with two interfaces: **{_vals[1]:+.0f} meV/atom**."),
    ])
    return


if __name__ == "__main__":
    app.run()
