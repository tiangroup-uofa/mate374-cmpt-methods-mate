# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.11", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import brentq

    return brentq, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    t_tc = mo.ui.slider(0.5, 1.3, step=0.02, value=0.8, label="T / Tc (regular solution)", show_value=True)
    c0 = mo.ui.slider(0.02, 0.5, step=0.01, value=0.2, label="alloy composition c₀", show_value=True)
    mo.hstack([t_tc, c0], justify="start", wrap=True)
    return c0, t_tc


@app.cell
def _(brentq, np):
    def g(c, t):
        """Regular-solution free energy per atom in units of k Tc, with Omega = 2 k Tc."""
        return 2 * c * (1 - c) + t * (c * np.log(c) + (1 - c) * np.log(1 - c))

    def tangent_points(t):
        """Common tangent of the symmetric curve: g'(c) = 0, the A2 equation F(x) = 0."""
        if t >= 1:
            return None
        F = lambda c: t * np.log(c / (1 - c)) + 2 * (1 - 2 * c)
        ca = brentq(F, 1e-12, 0.5 - np.sqrt(1 - t) / 2 - 1e-12)
        return ca, 1 - ca

    def spinodal(t):
        """Where g''(c) = 0: c (1 - c) = t / 4."""
        if t >= 1:
            return None
        return (1 - np.sqrt(1 - t)) / 2, (1 + np.sqrt(1 - t)) / 2

    return g, spinodal, tangent_points


@app.cell(hide_code=True)
def _(c0, g, mo, np, plt, spinodal, t_tc, tangent_points):
    _t, _c0 = t_tc.value, c0.value
    _cs = np.linspace(0.002, 0.998, 500)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    _a1.plot(_cs, g(_cs, _t), color="#231f20", lw=1.6)
    _tp, _sp = tangent_points(_t), spinodal(_t)
    if _tp:
        _a, _b = _tp
        _a1.plot([0, 1], [g(_a, _t)] * 2, "--", color="#b5473a", lw=1)
        _a1.plot([_a, _b], [g(_a, _t), g(_b, _t)], "o", color="#b5473a")
        _a1.axvspan(*_sp, color="#f5d9d9", alpha=0.6, lw=0)
    _a1.plot([_c0], [g(_c0, _t)], "s", color="#1f77b4", ms=7)
    _a1.set(xlabel="Ag fraction, c", ylabel="g / k T$_c$", title=f"free energy at T/Tc = {_t:.2f}")
    _a1.spines[["top", "right"]].set_visible(False)
    _ts = np.linspace(0.5, 0.999, 200)
    _bl = np.array([tangent_points(x)[0] for x in _ts])
    _sl = np.array([spinodal(x)[0] for x in _ts])
    _a2.fill_betweenx(_ts, _bl, 1 - _bl, color="#cde5ec")
    _a2.fill_betweenx(_ts, _sl, 1 - _sl, color="#f5d9d9")
    _a2.plot(np.r_[_bl, 0.5, (1 - _bl)[::-1]], np.r_[_ts, 1, _ts[::-1]], color="#1f77b4", lw=1.4)
    _a2.plot(np.r_[_sl, 0.5, (1 - _sl)[::-1]], np.r_[_ts, 1, _ts[::-1]], color="#b5473a", lw=1, ls="--")
    _a2.plot([_c0], [_t], "s", color="#1f77b4", ms=7)
    _a2.text(0.5, 0.62, "spinodal", ha="center", fontsize=9, color="#b5473a")
    _a2.text(0.12, 0.93, "nucleation", ha="center", fontsize=8, color="#1f77b4", rotation=80)
    _a2.text(0.5, 1.18, "one mixed phase", ha="center", fontsize=9)
    _a2.set(xlim=(0, 1), ylim=(0.5, 1.3), xlabel="Ag fraction, c", ylabel="T / Tc")
    _a2.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    if _tp is None:
        _msg = "Above Tc: one minimum, the alloy stays **mixed**."
    elif _sp[0] < _c0 < _sp[1]:
        _msg = f"Inside the spinodal (g'' < 0): every small fluctuation grows. **Spinodal decomposition** into {_tp[0]:.3f} and {_tp[1]:.3f}."
    elif _tp[0] < _c0 < _tp[1]:
        _msg = f"Between the tangent point and the spinodal: small fluctuations raise g. **Nucleation** of particles at {_tp[1]:.3f}."
    else:
        _msg = "Outside the tangent points: the uniform alloy is stable, **mixed**."
    mo.vstack([_fig, mo.md(_msg)])
    return


if __name__ == "__main__":
    app.run()
