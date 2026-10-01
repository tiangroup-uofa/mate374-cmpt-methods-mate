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
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    return minimize, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L14 · How different minimizers walk across one energy landscape

    A middle atom $M=(x,y)$ is joined by two harmonic springs to atoms fixed at
    $L=(-a,0)$ and $R=(a,0)$. Each spring has stiffness $k$ and natural length $l_0$:

    $$
    E(x,y)=\frac{k}{2}\left(r_L-l_0\right)^2+\frac{k}{2}\left(r_R-l_0\right)^2,
    \qquad r_L=\sqrt{(x+a)^2+y^2},\quad r_R=\sqrt{(x-a)^2+y^2}.
    $$

    Every spring is harmonic, yet $E$ is not quadratic in $x$ and $y$, because the
    spring lengths are square roots of the coordinates. With $l_0>a$, the springs are
    compressed when $M$ sits between $L$ and $R$, and the minima lie at
    $(0,\pm\sqrt{l_0^2-a^2})$, where both springs reach their natural length.

    Each **Live step** cell holds one part of the calculation. Choose a starting point
    and compare the paths that different methods take to a stationary point.
    We use $a=1$ and $k=1$.
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    l0_slider = mo.ui.slider(1.05, 1.5, value=1.2, step=0.05, label="Natural length l₀ (a = 1)",
                             show_value=True)
    x0_slider = mo.ui.slider(-1.5, 1.5, value=0.3, step=0.05, label="Start x₀", show_value=True)
    y0_slider = mo.ui.slider(-1.0, 1.5, value=0.1, step=0.05, label="Start y₀", show_value=True)
    mo.vstack([
        mo.hstack([l0_slider, x0_slider, y0_slider], justify="start", gap=2),
        mo.md("Start at (0.3, 0.1) first, then try (0.6, 0.4), (1.5, 0.05), and a smaller l₀."),
    ])
    return l0_slider, x0_slider, y0_slider


@app.cell(hide_code=True)
def step_one_text(mo):
    mo.md(r"""
    ### Live step 1 · Energy and gradient

    The gradient adds one force-like term per spring. For the spring to $L$, with
    $\mathbf d_L=(x+a,\,y)$ and $r_L=\lvert\mathbf d_L\rvert$,

    $$
    \nabla E=k\,(r_L-l_0)\,\frac{\mathbf d_L}{r_L}+k\,(r_R-l_0)\,\frac{\mathbf d_R}{r_R}.
    $$

    A stretched spring ($r>l_0$) pulls $M$ toward the fixed atom, and a compressed one
    pushes it away. The force on $M$ is $-\nabla E$.
    """)
    return


@app.cell
def live_energy(np):
    # Live step 1: energy and gradient of the middle atom, p = (x, y).
    A, K = 1.0, 1.0
    FIXED = np.array([[-A, 0.0], [A, 0.0]])   # positions of L and R

    def energy(p, l0):
        r = np.linalg.norm(p - FIXED, axis=1)
        return 0.5*K*np.sum((r - l0)**2)

    def gradient(p, l0):
        d = p - FIXED                          # (2, 2): one row per spring
        r = np.linalg.norm(d, axis=1)
        return np.sum((K*(r - l0)/r)[:, None]*d, axis=0)

    return FIXED, energy, gradient


@app.cell(hide_code=True)
def supplied_hessian(FIXED, np):
    def hessian(p, l0, k=1.0):
        """Analytical 2x2 Hessian: each spring adds k[u u^T + (1 - l0/r)(I - u u^T)]."""
        H = np.zeros((2, 2))
        for d in p - FIXED:
            r = np.linalg.norm(d)
            u = np.outer(d, d)/r**2
            H += k*(u + (1 - l0/r)*(np.eye(2) - u))
        return H

    return (hessian,)


@app.cell(hide_code=True)
def step_two_text(mo):
    mo.md(r"""
    ### Live step 2 · Gradient descent and Newton's method by hand

    **Gradient descent** steps a fixed distance downhill:
    $\mathbf x_{n+1}=\mathbf x_n-\alpha\,\nabla E(\mathbf x_n)$.

    **Newton's method** fits a paraboloid using the Hessian $\mathbf H$ and jumps to its
    stationary point. Each step solves the linear system from L13,

    $$
    \mathbf H(\mathbf x_n)\,\Delta\mathbf x=-\nabla E(\mathbf x_n),
    \qquad \mathbf x_{n+1}=\mathbf x_n+\Delta\mathbf x .
    $$

    Both loops record every point they visit so that we can draw the path.
    """)
    return


@app.cell
def live_hand_methods(gradient, hessian, np):
    # Live step 2: gradient descent and Newton iterations; each returns the visited points.
    def gradient_descent(p0, l0, step=0.4, tol=1e-6, max_iter=2000):
        path = [np.array(p0, dtype=float)]
        while np.linalg.norm(gradient(path[-1], l0)) > tol and len(path) <= max_iter:
            path.append(path[-1] - step*gradient(path[-1], l0))
        return np.array(path)

    def newton(p0, l0, tol=1e-10, max_iter=50):
        path = [np.array(p0, dtype=float)]
        while np.linalg.norm(gradient(path[-1], l0)) > tol and len(path) <= max_iter:
            dp = np.linalg.solve(hessian(path[-1], l0), -gradient(path[-1], l0))
            path.append(path[-1] + dp)
        return np.array(path)

    return gradient_descent, newton


@app.cell(hide_code=True)
def step_three_text(mo):
    mo.md(r"""
    ### Live step 3 · SciPy's minimizers with a callback

    `minimize` calls `callback(xk)` after every iteration. Appending `xk` to a list
    records the path. Nelder–Mead uses only energies, CG and BFGS also use the
    gradient, and L-BFGS-B keeps only a short history of gradients.

    ```python
    result = minimize(fun, x0, args=(l0,), jac=grad, method="BFGS", callback=record)
    result.x, result.nit, result.nfev, result.njev
    ```

    Docs: [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    """)
    return


@app.cell
def live_scipy(energy, gradient, minimize, np):
    # Live step 3: run one SciPy method and return its path and result.
    def scipy_path(p0, l0, method):
        path = [np.array(p0, dtype=float)]
        record = lambda xk, *args: path.append(np.array(xk, dtype=float))
        result = minimize(energy, p0, args=(l0,), method=method,
                          jac=None if method == "Nelder-Mead" else gradient,
                          callback=record, options={"maxiter": 2000})
        return np.array(path), result

    return (scipy_path,)


@app.cell(hide_code=True)
def run_methods(energy, gradient, gradient_descent, hessian, l0_slider, newton, np, scipy_path, x0_slider, y0_slider):
    l0 = l0_slider.value
    p_start = np.array([x0_slider.value, y0_slider.value])
    paths, rows = {}, []

    def classify_point(p):
        values = np.linalg.eigvalsh(hessian(p, l0))
        return "minimum" if values.min() > 0 else "saddle" if values.max() > 0 else "maximum"

    for name, run in [("Gradient descent", lambda: gradient_descent(p_start, l0)),
                      ("Newton", lambda: newton(p_start, l0))]:
        path = run()
        paths[name] = path
        rows.append((name, len(path) - 1, "—", path[-1]))
    for method in ["Nelder-Mead", "CG", "BFGS", "L-BFGS-B"]:
        path, result = scipy_path(p_start, l0, method)
        paths[method] = path
        rows.append((method, result.nit, result.nfev, path[-1]))
    method_table = "\n".join(
        ["| Method | Iterations | Energy evaluations | Final (x, y) | E | Stationary point |",
         "|---|---:|---:|---|---:|---|"]
        + [f"| {n} | {it} | {nf} | ({p[0]:+.4f}, {p[1]:+.4f}) | {energy(p, l0):.2e} | "
           f"{classify_point(p) if np.linalg.norm(gradient(p, l0)) < 1e-4 else 'not converged'} |"
           for n, it, nf, p in rows])
    return l0, method_table, p_start, paths


@app.cell(hide_code=True)
def plot_paths(energy, l0, method_table, mo, np, p_start, paths, plt):
    def draw():
        xs, ys = np.linspace(-1.6, 1.6, 241), np.linspace(-1.1, 1.6, 221)
        Xg, Yg = np.meshgrid(xs, ys)
        Z = np.array([[energy(np.array([x, y]), l0) for x in xs] for y in ys])
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.contourf(Xg, Yg, np.log10(Z + 1e-3), levels=30, cmap="Greys_r", alpha=0.6)
        ax.contour(Xg, Yg, Z, levels=np.linspace(0, 1.2, 13), colors="white", linewidths=0.5)
        colours = {"Gradient descent": "tab:blue", "Newton": "tab:red", "Nelder-Mead": "tab:purple",
                   "CG": "tab:olive", "BFGS": "tab:orange", "L-BFGS-B": "tab:cyan"}
        for name, path in paths.items():
            ax.plot(path[:, 0], path[:, 1], "o-", ms=3, lw=1.2, color=colours[name],
                    label=f"{name} ({len(path) - 1} steps)")
        ax.plot([-1, 1], [0, 0], "s", color="black", ms=10, label="Fixed atoms L, R")
        ax.plot(*p_start, "*", color="gold", mec="black", ms=16, label="Start")
        y_min = np.sqrt(max(l0**2 - 1, 0))
        ax.plot([0, 0], [y_min, -y_min], "x", color="green", ms=10, mew=2, label="Minima")
        ax.plot([0], [0], "P", color="tab:red", mec="black", ms=9, label="Saddle point (0, 0)")
        ax.set(xlabel="x (units of a)", ylabel="y (units of a)", aspect="equal",
               xlim=(-1.6, 1.6), ylim=(-1.1, 1.6),
               title=f"Paths on E(x, y), l₀ = {l0:.2f}a (shading: log₁₀ E)")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        fig.tight_layout()
        return fig

    paths_figure = draw()
    mo.vstack([paths_figure, mo.md(method_table), mo.md(
        "A zero gradient only identifies a **stationary point**. The last column uses the "
        "eigenvalues of the 2×2 Hessian, which L15 explains, to say which kind it is.")])
    return (paths_figure,)


if __name__ == "__main__":
    app.run()
