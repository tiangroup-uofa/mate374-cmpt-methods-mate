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
    from scipy.linalg import LinAlgError, cho_factor
    from scipy.optimize import minimize

    return LinAlgError, cho_factor, minimize, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L16 · The compressed spring: a saddle point and its Hessian

    Two atoms are fixed at $\mp a\,\mathbf n$, where $\mathbf n=(\cos\theta,\sin\theta)$
    is the direction of the line joining them. A middle atom $M=(x,y)$ is joined to each
    by a spring of stiffness $k$ and natural length $l_0>a$, so both springs are
    compressed when $M$ sits at the origin. Rotating the system by $\theta$ changes
    nothing physical, but it changes the entries of the Hessian in our fixed $x,y$
    coordinates. We use $a=1$ and $k=1$.

    **Predict:** rotate the geometry from 0° to 45°. Should the minima and eigenvectors
    rotate? Should the eigenvalues change? Compare both the matrix and energy profiles.
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    theta_slider = mo.ui.slider(0, 90, value=45, step=5, label="Rotation θ (degrees)", show_value=True)
    l0_slider = mo.ui.slider(1.05, 1.5, value=1.2, step=0.05, label="Natural length l₀ (a = 1)",
                             show_value=True)
    mo.hstack([theta_slider, l0_slider], justify="start", gap=2)
    return l0_slider, theta_slider


@app.cell(hide_code=True)
def supplied_springs(np):
    def fixed_atoms(theta_deg, a=1.0):
        n = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])
        return np.array([-a*n, a*n])

    def energy(p, fixed, l0, k=1.0):
        r = np.linalg.norm(p - fixed, axis=1)
        return 0.5*k*np.sum((r - l0)**2)

    def gradient(p, fixed, l0, k=1.0):
        d = p - fixed
        r = np.linalg.norm(d, axis=1)
        return np.sum((k*(r - l0)/r)[:, None]*d, axis=0)

    return energy, fixed_atoms, gradient


@app.cell(hide_code=True)
def step_one_text(mo):
    mo.md(r"""
    ### Live step 1 · The Hessian from the gradient

    Column $j$ of the Hessian is the change in the gradient when coordinate $j$ moves.
    A central difference builds it one column at a time:

    $$
    \mathbf H_{:,j}\approx\frac{\nabla E(\mathbf x+h\,\mathbf e_j)-\nabla E(\mathbf x-h\,\mathbf e_j)}{2h}.
    $$

    The analytical result at the origin is
    $\mathbf H=2k\left[\left(1-\tfrac{l_0}{a}\right)\mathbf I+\tfrac{l_0}{a}\,\mathbf n\mathbf n^{\mathsf T}\right]$,
    which the check below compares with the numerical matrix.
    """)
    return


@app.cell
def live_hessian(gradient, np):
    # Live step 1: finite-difference Hessian, made exactly symmetric.
    def hessian(p, fixed, l0, h=1e-5):
        n = len(p)
        H = np.empty((n, n))
        for j in range(n):
            step = np.zeros(n)
            step[j] = h
            H[:, j] = (gradient(p + step, fixed, l0) - gradient(p - step, fixed, l0))/(2*h)
        return 0.5*(H + H.T)

    return (hessian,)


@app.cell(hide_code=True)
def step_two_text(mo):
    mo.md(r"""
    ### Live step 2 · Eigenvalues and eigenvectors with `np.linalg.eigh`

    For a symmetric matrix, `eigh` returns the eigenvalues in ascending order and the
    unit eigenvectors as the **columns** of a matrix: `vectors[:, 0]` belongs to
    `values[0]`. Each pair satisfies $\mathbf H\mathbf v=\lambda\mathbf v$.

    ```python
    values, vectors = np.linalg.eigh(H)
    ```

    Docs: [`numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)
    """)
    return


@app.cell
def live_eigen(np):
    # Live step 2: eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix.
    def curvature_directions(H):
        values, vectors = np.linalg.eigh(H)
        return values, vectors

    return (curvature_directions,)


@app.cell(hide_code=True)
def analyse(curvature_directions, fixed_atoms, hessian, l0_slider, np, theta_slider):
    theta, l0 = theta_slider.value, l0_slider.value
    fixed = fixed_atoms(theta)
    origin = np.zeros(2)
    H0 = hessian(origin, fixed, l0)
    n_vec = fixed[1]
    H_exact = 2*((1 - l0)*np.eye(2) + l0*np.outer(n_vec, n_vec))
    values, vectors = curvature_directions(H0)
    residual = np.abs(H0 @ vectors - vectors*values).max()
    p_min = np.sqrt(l0**2 - 1)*np.array([-n_vec[1], n_vec[0]])   # one of the two minima
    H_min = hessian(p_min, fixed, l0)
    return H0, H_exact, H_min, fixed, l0, origin, p_min, residual, theta, values, vectors


@app.cell
def rotated_relaxation(energy, fixed, gradient, l0, minimize, mo, np, p_min):
    # A starting point close to one rotated minimum, with a small perturbation.
    rotated_result = minimize(energy, p_min + np.array([0.1, 0.07]),
                              args=(fixed, l0), jac=gradient, method="BFGS",
                              options={"gtol": 1e-10})
    mo.md(f"""
    **Relaxation in the rotated geometry:** position {np.round(rotated_result.x, 6)},
    energy {rotated_result.fun:.3e}, gradient norm
    {np.linalg.norm(gradient(rotated_result.x, fixed, l0)):.3e}.
    Distance from the geometric minimum: {np.linalg.norm(rotated_result.x - p_min):.3e}.
    """)
    return (rotated_result,)


@app.cell(hide_code=True)
def show_matrix(H0, H_exact, mo, np, residual, values, vectors):
    def fmt(M):
        return r"\begin{bmatrix}" + r"\\".join(" & ".join(f"{v:+.3f}" for v in row) for row in M) + r"\end{bmatrix}"

    mo.md(rf"""
    ### Hessian at the origin

    $$
    \mathbf H={fmt(H0)}\qquad
    \lambda_1={values[0]:+.3f},\ \mathbf v_1=({vectors[0, 0]:+.3f},{vectors[1, 0]:+.3f}),\qquad
    \lambda_2={values[1]:+.3f},\ \mathbf v_2=({vectors[0, 1]:+.3f},{vectors[1, 1]:+.3f})
    $$

    Diagonal entries $\partial^2E/\partial x^2={H0[0, 0]:+.3f}$ and
    $\partial^2E/\partial y^2={H0[1, 1]:+.3f}$. Largest difference from the analytical
    matrix: {np.abs(H0 - H_exact).max():.1e}. Largest entry of
    $\mathbf H\mathbf V-\mathbf V\boldsymbol\Lambda$: {residual:.1e}.
    """)
    return


@app.cell(hide_code=True)
def plot_landscape(energy, fixed, l0, mo, np, origin, p_min, plt, theta, values, vectors):
    def draw():
        xs = np.linspace(-1.5, 1.5, 241)
        Xg, Yg = np.meshgrid(xs, xs)
        Z = np.array([[energy(np.array([x, y]), fixed, l0) for x in xs] for y in xs])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
        ax1.contourf(Xg, Yg, np.log10(Z + 1e-3), levels=30, cmap="Greys_r", alpha=0.6)
        ax1.contour(Xg, Yg, Z, levels=np.linspace(0, 1.0, 11), colors="white", linewidths=0.5)
        ax1.plot(*fixed.T, "s", color="black", ms=10, label="Fixed atoms")
        ax1.plot(*p_min, "x", color="green", ms=10, mew=2, label="A minimum")
        ax1.plot(*(-p_min), "x", color="green", ms=10, mew=2)
        for lam, v, colour in zip(values, vectors.T, ["tab:red", "tab:blue"]):
            ax1.annotate("", xy=0.6*v, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=colour, lw=2.5))
            ax1.plot([], [], color=colour, lw=2.5, label=f"Eigenvector, λ = {lam:+.2f}")
        ax1.set(xlabel="x", ylabel="y", aspect="equal", title=f"E(x, y), θ = {theta}°, l₀ = {l0:.2f}a")
        ax1.legend(fontsize=7, loc="lower right")
        s = np.linspace(-0.6, 0.6, 201)
        for direction, label, style in [(np.array([1.0, 0.0]), "along x", "k--"),
                                        (np.array([0.0, 1.0]), "along y", "k:"),
                                        (vectors[:, 0], "along v₁", "tab:red"),
                                        (vectors[:, 1], "along v₂", "tab:blue")]:
            ax2.plot(s, [energy(origin + si*direction, fixed, l0) for si in s], style, label=label)
        ax2.set(xlabel="Displacement s from the origin", ylabel="E",
                title="Energy along four straight lines")
        ax2.legend(fontsize=8)
        for ax in (ax1, ax2):
            ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    landscape_figure = draw()
    mo.vstack([landscape_figure, mo.md(
        "Along x and y the energy can rise in both directions while the point is still a "
        "saddle. Along the eigenvector with the negative eigenvalue, it always falls.")])
    return (landscape_figure,)


@app.cell(hide_code=True)
def step_three_text(mo):
    mo.md(r"""
    ### Live step 3 · A Cholesky test for a minimum

    In L13, Cholesky factorization $\mathbf H=\mathbf L\mathbf L^{\mathsf T}$ required a
    **positive definite** matrix, one with all eigenvalues positive. Because the outer
    atoms are fixed here, there are no rigid translations or rotations, so a Hessian that
    passes this test at a stationary point establishes a strict local minimum.
    `cho_factor` raises `LinAlgError` otherwise.
    """)
    return


@app.cell
def live_cholesky(LinAlgError, cho_factor):
    # Live step 3: True if the symmetric matrix H is positive definite.
    def is_positive_definite(H):
        try:
            cho_factor(H)
            return True
        except LinAlgError:
            return False

    return (is_positive_definite,)


@app.cell(hide_code=True)
def show_cholesky(H0, H_min, is_positive_definite, mo, np, p_min):
    mo.md(f"""
    | Point | Eigenvalues | Positive definite (Cholesky) |
    |---|---|---|
    | Origin (0, 0) | {np.round(np.linalg.eigvalsh(H0), 3)} | {is_positive_definite(H0)} |
    | Minimum ({p_min[0]:+.3f}, {p_min[1]:+.3f}) | {np.round(np.linalg.eigvalsh(H_min), 3)} | {is_positive_definite(H_min)} |
    """)
    return


if __name__ == "__main__":
    app.run()
