# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## Two equations, one Newton step

    Before applying Newton to atomic forces, find the intersections of
    $F_1=x_2-\cosh(x_1/2)=0$ and $F_2=9x_1^2+25x_2^2-225=0$.
    Both residuals must vanish at the **same** point.

    **Predict:** which root will a positive or negative starting $x_1$ reach?
    What happens if you start exactly on $x_1=0$?
    """)
    return


@app.cell
def equations(np):
    def residual(x):
        x1, x2 = x
        return np.array([x2 - np.cosh(x1/2), 9*x1**2 + 25*x2**2 - 225])

    def jacobian(x):
        x1, x2 = x
        return np.array([[-0.5*np.sinh(x1/2), 1.0], [18*x1, 50*x2]])

    return jacobian, residual


@app.cell
def newton_algorithm(np):
    def newton_system(F, J, x0, ftol=1e-9, xtol=1e-10, max_steps=30):
        x = np.array(x0, dtype=float)
        history = []
        for m in range(max_steps + 1):
            f = F(x)
            matrix = J(x)
            if not (np.all(np.isfinite(f)) and np.all(np.isfinite(matrix))):
                return x, history, False, "Non-finite residual or Jacobian."
            try:
                delta = np.linalg.solve(matrix, -f)
            except np.linalg.LinAlgError:
                history.append((x.copy(), f, matrix, None))
                return x, history, False, "Singular Jacobian: try an off-axis starting point."
            history.append((x.copy(), f, matrix, delta))
            if np.max(np.abs(f)) < ftol and np.max(np.abs(delta)) < xtol:
                return x, history, True, "Converged: both residual and correction are small."
            if m == max_steps:
                break
            x = x + delta
            if np.max(np.abs(x)) > 100:
                return history[-1][0], history, False, "Step left the plotting domain by a large amount. Try a closer guess."
        return x, history, False, "Maximum number of Newton steps reached."

    return (newton_system,)


@app.cell(hide_code=True)
def controls(mo):
    guess1 = mo.ui.number(-4.0, 4.0, value=2.0, step=0.1, label="Initial x₁")
    guess2 = mo.ui.number(-4.0, 4.0, value=2.0, step=0.1, label="Initial x₂")
    mo.hstack([guess1, guess2], justify="start", gap=2)
    return guess1, guess2


@app.cell(hide_code=True)
def calculation(guess1, guess2, jacobian, newton_system, np, residual):
    solution, history, converged, status = newton_system(
        residual, jacobian, [guess1.value, guess2.value])
    paths = np.array([row[0] for row in history])
    reference_roots = np.array([
        newton_system(residual, jacobian, [sign*2.0, 2.0])[0] for sign in (-1, 1)
    ])
    return converged, history, paths, reference_roots, solution, status


@app.cell(hide_code=True)
def geometry(np, paths, plt, reference_roots):
    _u, _v = np.meshgrid(np.linspace(-5.3, 5.3, 110), np.linspace(-3.3, 4.2, 100))
    _f1 = _v - np.cosh(_u/2)
    _f2 = (9*_u**2 + 25*_v**2 - 225)/75
    system_figure = plt.figure(figsize=(10, 4.5), layout="constrained")
    _ax3 = system_figure.add_subplot(121, projection="3d")
    _ax2 = system_figure.add_subplot(122)
    _ax3.plot_surface(_u, _v, _f1, color="#1f77b4", alpha=0.28, linewidth=0)
    _ax3.plot_surface(_u, _v, _f2, color="#b5473a", alpha=0.28, linewidth=0)
    _ax3.contour(_u, _v, _f1, levels=[0], colors=["#1f77b4"], linewidths=2)
    _ax3.contour(_u, _v, _f2, levels=[0], colors=["#b5473a"], linewidths=2)
    _ax3.scatter(reference_roots[:, 0], reference_roots[:, 1], [0, 0], c="black", s=35)
    _ax3.set(xlabel="x₁", ylabel="x₂", zlabel="F₁ or F₂/75", title="Residual surfaces and their zero curves")
    _ax3.view_init(elev=26, azim=-65)
    _ax2.contour(_u, _v, _f1, levels=[0], colors=["#1f77b4"], linewidths=2)
    _ax2.contour(_u, _v, _f2, levels=[0], colors=["#b5473a"], linewidths=2)
    _ax2.plot([], [], color="#1f77b4", label="F₁ = 0: x₂ = cosh(x₁/2)")
    _ax2.plot([], [], color="#b5473a", label="F₂ = 0: ellipse")
    _ax2.plot(paths[:, 0], paths[:, 1], "o--", color="#231f20", ms=4, label="Newton iterates")
    _ax2.plot(paths[0, 0], paths[0, 1], "s", color="#231f20", ms=7, label="Initial guess")
    _ax2.scatter(reference_roots[:, 0], reference_roots[:, 1], c="black", marker="*", s=100)
    _ax2.set(xlim=(-5.4, 5.4), ylim=(-3.4, 4.3), xlabel="x₁", ylabel="x₂", title="Projection onto the coordinate plane")
    _ax2.legend(fontsize=8, loc="lower center")
    system_figure
    return (system_figure,)


@app.cell(hide_code=True)
def iteration_control(history, mo):
    iteration = mo.ui.slider(0, max(1, len(history)-1), value=0, step=1,
                             label="Inspect Newton iteration m", show_value=True, full_width=True)
    return (iteration,)


@app.cell(hide_code=True)
def iteration_detail(history, iteration, mo, np):
    _m = min(iteration.value, len(history)-1)
    _x, _f, _j, _delta = history[_m]
    def tex_matrix(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[:, None]
        return r"\begin{bmatrix}" + r"\\".join(" & ".join(f"{v:.5g}" for v in row) for row in a) + r"\end{bmatrix}"
    _equation = (r"\text{Singular Jacobian: no unique correction.}" if _delta is None
                 else tex_matrix(_j) + tex_matrix(_delta) + "=-" + tex_matrix(_f))
    mo.vstack([
        mo.md(rf"""
        $$\mathbf J\,\Delta\mathbf x=-\mathbf F$$

        $$ {_equation} $$
        """),
        iteration,
    ])
    return


@app.cell(hide_code=True)
def convergence_table(history, mo, np):
    mo.ui.table([
        {"m": m, "x₁": f"{row[0][0]:.6f}", "x₂": f"{row[0][1]:.6f}",
         "max |Fᵢ|": f"{np.max(np.abs(row[1])):.3e}",
         "max |Δxᵢ|": "—" if row[3] is None else f"{np.max(np.abs(row[3])):.3e}"}
        for m, row in enumerate(history)
    ], selection=None)
    return


@app.cell(hide_code=True)
def interpretation(mo):
    mo.md(r"""
    **Try:** change the initial guess from $(2,2)$ to $(-2,2)$ and then $(0,2)$.
    The two roots are approximately $(\pm3.03116,2.38587)$. At $x_1=0$, the first
    column of the Jacobian is zero, so the linear solve cannot determine a unique step.
    For force balance, the same calculation will use atomic coordinates and net forces
    in place of these two test functions.
    """)
    return


if __name__ == "__main__":
    app.run()
