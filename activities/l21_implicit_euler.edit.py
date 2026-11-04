# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    return mo, np, plt, solve_ivp


@app.cell
def _(np, solve_ivp):
    # A steel sphere (radius 1 mm) cooling by radiation alone in a vacuum furnace
    SIGMA_SB = 5.670374419e-8      # W m^-2 K^-4
    EMISSIVITY, RHO, CP, RADIUS = 0.8, 7800.0, 500.0, 1.0e-3   # -, kg/m^3, J/(kg K), m
    T_AMB, T_START, T_END = 300.0, 1200.0, 60.0                # K, K, s
    alpha = 3 * EMISSIVITY * SIGMA_SB / (RHO * CP * RADIUS)    # K^-3 s^-1


    def rhs(t, T):
        return -alpha * (T**4 - T_AMB**4)


    reference = solve_ivp(rhs, [0, T_END], [T_START], method="Radau", rtol=1e-11, atol=1e-9, dense_output=True)


    def explicit_euler(h):
        n = round(T_END / h)
        t = np.arange(n + 1) * h
        T = np.empty(n + 1)
        T[0] = T_START
        for i in range(n):
            T[i + 1] = T[i] + h * rhs(t[i], T[i])
        return t, T


    def implicit_euler(h):
        n = round(T_END / h)
        t = np.arange(n + 1) * h
        T = np.empty(n + 1)
        T[0] = T_START
        for i in range(n):
            x = T[i]                                           # starting guess for T[i+1]
            for _ in range(50):                                # Newton iteration on g(x) = 0
                g = x - T[i] - h * rhs(t[i + 1], x)
                dg = 1 + 4 * h * alpha * x**3
                x -= g / dg
                if abs(g / dg) < 1e-10:
                    break
            T[i + 1] = x
        return t, T

    return T_AMB, T_END, T_START, explicit_euler, implicit_euler, reference


@app.cell
def _(mo):
    method = mo.ui.dropdown(options=["explicit Euler", "implicit Euler"], value="implicit Euler", label="method")
    h_slider = mo.ui.slider(steps=[30.0, 20.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.5], value=20.0, label="step h (s)", show_value=True)
    return h_slider, method


@app.cell
def _(T_AMB, T_END, T_START, explicit_euler, h_slider, implicit_euler, method, mo, np, plt, reference):
    h = h_slider.value
    t, T = (explicit_euler if method.value == "explicit Euler" else implicit_euler)(h)
    T_ref = reference.sol(t)[0]
    err = T - T_ref

    rows = ["| t (s) | " + method.value + " T (K) | reference T (K) | error (K) |", "|---|---|---|---|"]
    for _t in [10, 20, 30, 60]:
        _i = int(np.argmin(abs(t - _t)))
        if abs(t[_i] - _t) < 1e-9:
            rows.append(f"| {_t} | {T[_i]:.1f} | {T_ref[_i]:.1f} | {err[_i]:+.1f} |")
    table = mo.md("\n".join(rows))

    fine = np.linspace(0, T_END, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    ax.plot(fine, reference.sol(fine)[0], color="0.55", lw=2.5, label="reference solution")
    ax.plot(t, T, "o-", color="C3" if method.value == "explicit Euler" else "C2", ms=4, lw=1.2, label=f"{method.value}, h = {h:g} s")
    ax.axhline(T_AMB, color="0.7", ls=":", lw=1)
    ax.set(xlabel="time (s)", ylabel="T (K)", ylim=(min(-100, T.min() - 50), T_START + 100))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    mo.vstack([mo.hstack([method, h_slider], justify="start", align="center", gap=2), table, fig])
    return


if __name__ == "__main__":
    app.run()

