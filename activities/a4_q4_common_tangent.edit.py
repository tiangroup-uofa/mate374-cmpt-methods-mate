# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.optimize import root

    return CubicSpline, mo, np, plt, root


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A4 Q4 · Common tangent for Pb–Sn solder at 500 K

    **Student notebook.** Complete the cells marked **Step 1**, **Step 2**, and
    **Step 3** in order. Each step has a check, and the figure at the end updates as
    you complete the steps.

    At 500 K, Pb–Sn alloys between the compositions $\alpha$ and $\beta$ separate into
    a Pb-rich fcc phase and a Sn-rich liquid. The cell below stores the molar Gibbs
    energy of each phase at $T=500$ K against the mole fraction of tin, $x_{\mathrm{Sn}}$:

    - `x_fcc` and `G_fcc_data`: compositions and Gibbs energies of the fcc phase;
    - `x_liq` and `G_liq_data`: compositions and Gibbs energies of the liquid.
    """)
    return


@app.cell
def supplied_data(np):
    # Molar Gibbs energies (J/mol) at 500 K, evaluated from the NIST solder
    # database (U. R. Kattner, NIST; Pb–Sn parameters of Fecht et al., 1989).
    # Reference states: pure fcc Pb at x_Sn = 0 and pure bct Sn at x_Sn = 1.
    x_fcc = np.round(np.arange(0.0, 0.501, 0.05), 2)
    G_fcc_data = np.array([0.0, -483.6, -695.9, -815.8, -881.1, -909.4,
                           -910.6, -891.2, -855.1, -805.2, -742.9])
    x_liq = np.round(np.arange(0.30, 0.951, 0.05), 2)
    G_liq_data = np.array([-726.3, -814.7, -887.0, -945.6, -991.4, -1024.8, -1045.2,
                           -1050.8, -1039.2, -1006.2, -945.6, -847.3, -693.9, -447.5])
    return G_fcc_data, G_liq_data, x_fcc, x_liq


@app.cell(hide_code=True)
def data_table(G_fcc_data, G_liq_data, mo, x_fcc, x_liq):
    def _table(x, G):
        return "| $x_{\\mathrm{Sn}}$ | $G$ (J/mol) |\n|---:|---:|\n" + "\n".join(
            f"| {xi:g} | {Gi:.1f} |" for xi, Gi in zip(x, G))

    mo.vstack([
        mo.md(
            "Data source: [NIST solder database, Pb–Sn system]"
            "(https://www.metallurgy.nist.gov/phase/solder/pbsn.html)."),
        mo.hstack([mo.md("**fcc (Pb-rich):** `x_fcc`, `G_fcc_data`\n\n" + _table(x_fcc, G_fcc_data)),
                   mo.md("**liquid (Sn-rich):** `x_liq`, `G_liq_data`\n\n" + _table(x_liq, G_liq_data))],
                  justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def step_one(mo):
    mo.md(r"""
    ### **Step 1** · Interpolate each phase

    Build one cubic spline per phase, so that `G_fcc(x)` and `G_liq(x)` return
    the molar Gibbs energy at any composition inside each data range.

    ```python
    spline = CubicSpline(x_data, y_data)
    spline(0.123)          # value between the data points
    ```

    Docs: [`scipy.interpolate.CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
    """)
    return


@app.cell
def step_one_interpolate(CubicSpline, G_fcc_data, G_liq_data, x_fcc, x_liq):
    # Step 1: replace None with cubic splines of (x_fcc, G_fcc_data)
    # and (x_liq, G_liq_data).
    G_fcc = None
    G_liq = None
    return G_fcc, G_liq


@app.cell(hide_code=True)
def step_one_check(G_fcc, G_liq, mo, np):
    # Spline values between the data points, from the reference solution.
    try:
        _values = [float(G_fcc(0.125)), float(G_liq(0.625))]
        _ok = np.allclose(_values, [-763.06, -1049.98], atol=0.01)
    except Exception:
        _values, _ok = None, False
    interpolation_done = _ok
    mo.callout(
        mo.md(f"**Check passed:** $G_{{\\mathrm{{fcc}}}}(0.125)={_values[0]:.2f}$ J/mol and "
              f"$G_{{\\mathrm{{liq}}}}(0.625)={_values[1]:.2f}$ J/mol.") if _ok
        else mo.md("**Check not passed yet:** `G_fcc(0.125)` should give about −763.06 J/mol "
                   "and `G_liq(0.625)` about −1049.98 J/mol."),
        kind="success" if _ok else "warn")
    return (interpolation_done,)


@app.cell(hide_code=True)
def step_two(mo):
    mo.md(r"""
    ### **Step 2** · Slope of each curve

    Complete the two functions `slope_fcc(x)` and `slope_liq(x)`. Each one returns
    $\mathrm{d}G/\mathrm{d}x_{\mathrm{Sn}}$ (J/mol) of its spline at the composition `x`
    from the central difference

    $$
    G'(x)\approx\frac{G(x+h)-G(x-h)}{2h},
    $$

    with the step `h` defined in the cell. The function `central(f, x, h)` from
    [Lecture 18](https://tiangroup-uofa.github.io/mate374-cmpt-methods-mate/units/04/L18/index.html#derivatives-of-a-python-function-at-a-point)
    can be used directly.
    """)
    return


@app.cell
def step_two_slopes(G_fcc, G_liq, np):
    h = 1e-5  # step of the central difference


    def slope_fcc(x):
        # Step 2: replace np.nan with the central difference of G_fcc at x
        return np.nan


    def slope_liq(x):
        # Step 2: replace np.nan with the central difference of G_liq at x
        return np.nan

    return slope_fcc, slope_liq


@app.cell(hide_code=True)
def step_two_check(mo, np, slope_fcc, slope_liq):
    try:
        _values = [float(slope_fcc(0.2)), float(slope_liq(0.6))]
        _ok = np.allclose(_values, [-882.25, -266.75], atol=0.05)
    except Exception:
        _values, _ok = None, False
    slopes_done = _ok
    mo.callout(
        mo.md(f"**Check passed:** $G'_{{\\mathrm{{fcc}}}}(0.2)={_values[0]:.2f}$ J/mol and "
              f"$G'_{{\\mathrm{{liq}}}}(0.6)={_values[1]:.2f}$ J/mol.") if _ok
        else mo.md("**Check not passed yet:** `slope_fcc(0.2)` should give about −882 J/mol "
                   "and `slope_liq(0.6)` about −267 J/mol."),
        kind="success" if _ok else "warn")
    return (slopes_done,)


@app.cell(hide_code=True)
def step_three(mo):
    mo.md(r"""
    ### **Step 3** · Solve for the tangent points

    The common tangent touches the fcc curve at $\alpha$ and the liquid curve at
    $\beta$. The straight line joining the points $(\alpha, G_{\mathrm{fcc}}(\alpha))$ and
    $(\beta, G_{\mathrm{liq}}(\beta))$ has slope
    $[G_{\mathrm{liq}}(\beta)-G_{\mathrm{fcc}}(\alpha)]/(\beta-\alpha)$. For a common
    tangent, this slope must equal the slope of each curve at its tangent point:

    $$
    \begin{aligned}
    r_1(\alpha,\beta)&=G_{\mathrm{liq}}(\beta)-G_{\mathrm{fcc}}(\alpha)-G'_{\mathrm{fcc}}(\alpha)\,(\beta-\alpha)=0,\\
    r_2(\alpha,\beta)&=G_{\mathrm{liq}}(\beta)-G_{\mathrm{fcc}}(\alpha)-G'_{\mathrm{liq}}(\beta)\,(\beta-\alpha)=0.
    \end{aligned}
    $$

    Complete the two functions in the cell below:

    1. `tangent_residuals(p)` receives the vector `p = [alpha, beta]` and returns the
       two residuals `[r1, r2]`. Calculate `r1` from `G_fcc`, `G_liq`, and
       `slope_fcc`, and `r2` from `G_fcc`, `G_liq`, and `slope_liq`.
    2. `solve_alpha_beta(alpha_guess, beta_guess)` calls
       `root(tangent_residuals, [alpha_guess, beta_guess])` and returns `alpha, beta`
       from `result.x`.

    The sliders above the figure set the initial guesses for $\alpha$ and $\beta$.
    Start close to the minima of the two curves.

    Docs: [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)
    """)
    return


@app.cell
def step_three_solve(G_fcc, G_liq, np, root, slope_fcc, slope_liq):
    def tangent_residuals(p):
        alpha, beta = p
        # Step 3: replace np.nan with the two residuals
        r1 = np.nan
        r2 = np.nan
        return [r1, r2]


    def solve_alpha_beta(alpha_guess, beta_guess):
        # Step 3: call root on tangent_residuals from [alpha_guess, beta_guess]
        alpha, beta = np.nan, np.nan
        return alpha, beta

    return (solve_alpha_beta,)


@app.cell(hide_code=True)
def guesses(mo):
    alpha_guess = mo.ui.slider(0.05, 0.50, value=0.28, step=0.01,
                               show_value=True, label="Initial guess for α (fcc)")
    beta_guess = mo.ui.slider(0.30, 0.95, value=0.64, step=0.01,
                              show_value=True, label="Initial guess for β (liquid)")
    return alpha_guess, beta_guess


@app.cell(hide_code=True)
def figure(G_fcc, G_fcc_data, G_liq, G_liq_data, alpha_guess, beta_guess,
           interpolation_done, mo, np, plt, slope_fcc, slope_liq, slopes_done,
           solve_alpha_beta, x_fcc, x_liq):
    # Solve from the slider guesses and accept only a genuine common tangent:
    # small residuals, two distinct points, each inside its data range.
    try:
        alpha, beta = (float(_v) for _v in solve_alpha_beta(alpha_guess.value, beta_guess.value))
        _r1 = G_liq(beta) - G_fcc(alpha) - slope_fcc(alpha) * (beta - alpha)
        _r2 = G_liq(beta) - G_fcc(alpha) - slope_liq(beta) * (beta - alpha)
        solution_found = bool(
            slopes_done and np.all(np.abs([_r1, _r2]) < 1e-2)
            and x_fcc[0] <= alpha <= x_fcc[-1] and x_liq[0] <= beta <= x_liq[-1]
            and beta - alpha > 0.05)
    except Exception:
        alpha, beta, solution_found = np.nan, np.nan, False

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x_fcc, G_fcc_data, "o", ms=5, color="tab:blue", label="fcc (Pb-rich) data")
    ax.plot(x_liq, G_liq_data, "s", ms=5, color="tab:red", label="liquid (Sn-rich) data")
    if interpolation_done:
        _xa = np.linspace(x_fcc[0], x_fcc[-1], 300)
        _xb = np.linspace(x_liq[0], x_liq[-1], 300)
        ax.plot(_xa, G_fcc(_xa), color="tab:blue", lw=1.2)
        ax.plot(_xb, G_liq(_xb), color="tab:red", lw=1.2)
    for _x0, _color in [(alpha_guess.value, "tab:blue"), (beta_guess.value, "tab:red")]:
        ax.axvline(_x0, color=_color, lw=0.8, ls=":", alpha=0.6)
    if solution_found:
        tangent_slope = float(slope_fcc(alpha))
        _xt = np.array([0.0, 1.0])
        ax.plot(_xt, G_fcc(alpha) + tangent_slope * (_xt - alpha), "k--",
                lw=1, label="common tangent")
        for _x0, _G in [(alpha, G_fcc), (beta, G_liq)]:
            ax.vlines(_x0, -1300, _G(_x0), color="0.5", lw=0.8, ls=":")
            ax.plot(_x0, _G(_x0), "k*", ms=9)
    else:
        tangent_slope = np.nan
    ax.set_xlim(0, 1)
    ax.set_ylim(-1300, 100)
    ax.set_xlabel(r"$x_{\mathrm{Sn}}$")
    ax.set_ylabel("$G$ (J/mol)")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()

    def _mark(ok, done, todo):
        return f"✅ {done}" if ok else f"❌ {todo}"

    _status = mo.md(
        "| Interpolation | Solution | Tangent slope (J/mol) | α | β |\n"
        "|:-:|:-:|:-:|:-:|:-:|\n"
        f"| {_mark(interpolation_done, 'done', 'not done')} "
        f"| {_mark(solution_found, 'found', 'not found')} "
        + (f"| {tangent_slope:.1f} | {alpha:.4f} | {beta:.4f} |"
           if solution_found else "| – | – | – |"))
    mo.vstack([mo.hstack([alpha_guess, beta_guess], justify="start", gap=2),
               fig, _status], gap=0.5)
    return alpha, beta, fig, solution_found, tangent_slope


if __name__ == "__main__":
    app.run()
