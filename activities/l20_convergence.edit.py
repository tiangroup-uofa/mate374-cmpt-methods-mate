# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo, np):
    # name: (function, a, b, exact value, LaTeX of the integral)
    tests = {
        "exp(x) on [0, 1]": (np.exp, 0.0, 1.0, np.e - 1, r"\int_0^1 e^x\,\mathrm dx = e-1 = 1.718281828\ldots"),
        "sin(x) on [0, π/2]": (np.sin, 0.0, np.pi / 2, 1.0, r"\int_0^{\pi/2}\sin x\,\mathrm dx = 1"),
        "x³ on [1, 3]": (lambda x: x**3, 1.0, 3.0, 20.0, r"\int_1^3 x^3\,\mathrm dx = 20"),
    }
    choice = mo.ui.dropdown(options=list(tests), value="exp(x) on [0, 1]", label="integral")
    return choice, tests


@app.cell
def _(np):
    def composite(rule, f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        if rule == "left rectangle":
            return h * np.sum(f(x[:-1]))
        if rule == "midpoint":
            return h * np.sum(f((x[:-1] + x[1:]) / 2))
        fx = f(x)
        if rule == "trapezoid":
            return h * (fx[0] / 2 + np.sum(fx[1:-1]) + fx[-1] / 2)
        if rule == "Simpson 1/3":
            return h / 3 * (fx[0] + 4 * np.sum(fx[1:-1:2]) + 2 * np.sum(fx[2:-1:2]) + fx[-1])
        if rule == "Simpson 3/8":
            w = np.full(n + 1, 3.0)
            w[0] = w[-1] = 1.0
            w[3:-1:3] = 2.0
            return 3 * h / 8 * np.sum(w * fx)


    rules = ["left rectangle", "midpoint", "trapezoid", "Simpson 1/3", "Simpson 3/8"]
    return composite, rules


@app.cell
def _(choice, composite, mo, np, plt, rules, tests):
    f, a, b, exact, latex = tests[choice.value]
    ns = 6 * 2 ** np.arange(13)                  # N = 6, 12, ..., 24576: valid for every rule
    hs = (b - a) / ns
    errors = {_r: np.array([abs(exact - composite(_r, f, a, b, int(_n))) for _n in ns]) for _r in rules}

    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    colors = {"left rectangle": "C7", "midpoint": "C0", "trapezoid": "C3", "Simpson 1/3": "C2", "Simpson 3/8": "C4"}
    is_cubic = choice.value == "x³ on [1, 3]"
    for _r in rules:
        if is_cubic and _r.startswith("Simpson"):
            continue
        ax.loglog(hs, np.maximum(errors[_r], 1e-17), "o-", ms=3, lw=1.5, color=colors[_r])
        # Label beside the fine-grid end of each curve, before round-off dominates.
        if _r.startswith("Simpson"):
            _j = 5 if _r == "Simpson 1/3" else 4
            _dy = -8 if _r == "Simpson 1/3" else 8
        else:
            _j = len(hs) - 1
            _dy = {"left rectangle": 0, "midpoint": -7, "trapezoid": 7}[_r]
        ax.annotate(
            _r, (hs[_j], errors[_r][_j]), xytext=(9, _dy),
            textcoords="offset points", fontsize=12, color=colors[_r],
            ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1),
        )

    # Half-decade-wide wedges show the theoretical error orders.
    for _r, _k, _p in [("left rectangle", 3, 1), ("trapezoid", 3, 2), ("Simpson 1/3", 2, 4)]:
        if is_cubic and _p == 4:
            continue
        _h1, _h2 = hs[_k], hs[_k] / 10**0.5
        _e1 = errors[_r][_k] / 6
        _e2 = _e1 * (_h2 / _h1)**_p
        ax.plot([_h1, _h2, _h1, _h1], [_e1, _e2, _e2, _e1], color=colors[_r], lw=1.2)
        _order = r"$O(h)$" if _p == 1 else rf"$O(h^{_p})$"
        ax.text(_h1 * 1.25, np.sqrt(_e1 * _e2), _order,
                ha="right", va="center", fontsize=14, color=colors[_r])

    roundoff = np.finfo(float).eps * abs(exact)
    ax.axhline(roundoff, color="0.6", lw=0.8, ls="--")
    ax.text(hs[0], roundoff * 2, r"round-off scale $\epsilon |I|$",
            fontsize=11, color="0.4", ha="left")
    if is_cubic:
        ax.text(0.48, 0.2, "Both Simpson rules: exact for x³\n(round-off only)",
                transform=ax.transAxes, fontsize=13, color="0.2")
    ax.set_xlabel("subinterval width h", fontsize=14)
    ax.set_ylabel("|error|", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(1e-17, max(3, max(_e.max() for _e in errors.values()) * 2))
    ax.set_xlim(hs[0] * 1.5, hs[-1] / 15)
    ax.grid(alpha=0.2, which="major")

    mo.vstack([choice, mo.md(f"$$\n{latex}\n$$"), fig])
    plt.close(fig)
    return


if __name__ == "__main__":
    app.run()
