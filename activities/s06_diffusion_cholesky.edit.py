# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import time
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.fft import dstn, idstn
    return Rectangle, dstn, idstn, mo, np, plt, time


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## S06 · One 2D diffusion matrix, different source fields

    Take the five-point finite-difference matrix for steady diffusion on a square
    plate with fixed zero concentration on the boundary. Each interior grid point
    interacts with its left, right, lower, and upper neighbours, so the matrix is
    sparse, symmetric, and positive definite.

    **Predict:** if only the source field changes, which part of a direct solve
    could be reused? Then compare that idea with a solver that uses the special
    structure of this rectangular-grid diffusion problem.
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    side_form = mo.ui.dropdown(
        [10, 30, 100, 300, 1000], value=100,
        label="Interior grid points per side s",
    ).form(submit_button_label="Solve 2D diffusion problem")
    mo.vstack([side_form, mo.md(
        "The number of unknowns is n = s². The s = 1000 case has one million "
        "unknowns and is included to show why a dense matrix is the wrong model."
    )])
    return (side_form,)


@app.cell
def grid_size(side_form):
    side = int(side_form.value) if side_form.value is not None else 100
    N = side*side
    number_of_rhs = 2
    return N, number_of_rhs, side


@app.cell
def operation_estimates(N, np, number_of_rhs, side):
    dense_repeated_ops = number_of_rhs*((2/3)*N**3 + 2*N**2)
    dense_lu_ops = (2/3)*N**3 + number_of_rhs*2*N**2
    dense_cholesky_ops = (1/3)*N**3 + number_of_rhs*2*N**2
    # A separable rectangular-grid Poisson solve uses sine transforms in both directions.
    fast_poisson_work = number_of_rhs*N*np.log2(max(side, 2))
    one_tflop_seconds = {
        "dense_repeated": dense_repeated_ops/1e12,
        "dense_lu": dense_lu_ops/1e12,
        "dense_cholesky": dense_cholesky_ops/1e12,
        "fast_poisson": fast_poisson_work/1e12,
    }
    return dense_cholesky_ops, dense_lu_ops, dense_repeated_ops, fast_poisson_work, one_tflop_seconds


@app.cell(hide_code=True)
def show_estimates(N, dense_cholesky_ops, dense_lu_ops, dense_repeated_ops, fast_poisson_work, mo, number_of_rhs, one_tflop_seconds, side):
    mo.md(f"""
    ### Operation estimates: {side:,} × {side:,} grid = {N:,} unknowns

    | Approach | Approximate operations | Idealized time at 1 TFLOP/s |
    |---|---:|---:|
    | Fresh dense elimination for each source | {dense_repeated_ops:,.2e} | {one_tflop_seconds['dense_repeated']:,.2g} s |
    | Dense LU once, then reuse | {dense_lu_ops:,.2e} | {one_tflop_seconds['dense_lu']:,.2g} s |
    | Dense Cholesky once, then reuse | {dense_cholesky_ops:,.2e} | {one_tflop_seconds['dense_cholesky']:,.2g} s |
    | Optimized 2D diffusion solve used below | about {fast_poisson_work:,.2e} | {one_tflop_seconds['fast_poisson']:,.2g} s |

    Dense Cholesky is the best dense direct method in this list, but the optimized
    diffusion solve is in a different scaling class. It uses the grid structure
    directly instead of building the full dense matrix.
    """)
    return


@app.cell(hide_code=True)
def supplied_data(np, side):
    length = 1.0
    h = length/(side + 1)
    axis = h*np.arange(1, side + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    sources = np.stack([
        np.ones_like(X),
        1.0 + 0.75*np.exp(-((X - 0.30)**2 + (Y - 0.55)**2)/0.015),
    ], axis=-1)
    rhs = h*h*sources
    return X, Y, axis, h, length, rhs, sources


@app.cell(hide_code=True)
def solve_text(mo):
    mo.md(r"""
    ### The actual solve: diagonalize the 2D diffusion operator

    For a rectangular grid with fixed boundary values, the sine modes are the
    eigenvectors of the five-point diffusion matrix. The code below transforms
    each source field into sine-mode coefficients, divides by the eigenvalues, and
    transforms back. This is the kind of specialized solver that makes a
    million-unknown 2D diffusion calculation reasonable.
    """)
    return


@app.cell
def solve_2d_diffusion(dstn, idstn, np, rhs, side, time):
    t0 = time.perf_counter()
    modes = dstn(rhs, type=1, axes=(0, 1), norm="ortho")
    p = np.arange(1, side + 1)
    lam1 = 2 - 2*np.cos(np.pi*p/(side + 1))
    eigenvalues = lam1[:, None] + lam1[None, :]
    solution_modes = modes/eigenvalues[:, :, None]
    concentrations = idstn(solution_modes, type=1, axes=(0, 1), norm="ortho")
    solve_seconds = time.perf_counter() - t0
    return concentrations, eigenvalues, solve_seconds


@app.cell(hide_code=True)
def residual_check(concentrations, np, rhs):
    stencil = 4*concentrations.copy()
    stencil[1:, :, :] -= concentrations[:-1, :, :]
    stencil[:-1, :, :] -= concentrations[1:, :, :]
    stencil[:, 1:, :] -= concentrations[:, :-1, :]
    stencil[:, :-1, :] -= concentrations[:, 1:, :]
    residuals = stencil - rhs
    scaled_residuals = np.max(np.abs(residuals), axis=(0, 1))/(np.max(np.abs(rhs), axis=(0, 1)) + 4*np.max(np.abs(concentrations), axis=(0, 1)))
    return residuals, scaled_residuals


@app.cell(hide_code=True)
def results(N, concentrations, h, mo, np, residuals, rhs, scaled_residuals, side, solve_seconds):
    mo.md(f"""
    **Actual optimized solve time in this browser/Python session:** {solve_seconds:.3g} s.

    **Largest scaled residual, either source:** {np.max(scaled_residuals):.2e}.

    | Storage | Size |
    |---|---:|
    | Dense matrix, if allocated | {8*N*N/1e9:,.3f} GB |
    | Two source fields | {rhs.nbytes/1e6:.3f} MB |
    | Two solution fields | {concentrations.nbytes/1e6:.3f} MB |

    The dense matrix for s = 1000 would require 8 TB before any factorization.
    """)
    return


@app.cell(hide_code=True)
def profile_plot(X, Y, concentrations, mo, np, plt, side):
    mid = side//2
    _x = X[:, mid]
    with plt.rc_context({"font.size": 13}):
        profile_figure, _axes = plt.subplots(1, 2, figsize=(8.0, 3.5), layout="constrained")
        for _j, _name, _color in [(0, "Uniform source", "#231f20"),
                                  (1, "Localized stronger source", "#b5473a")]:
            _axes[0].plot(_x, concentrations[:, mid, _j], label=_name, color=_color, lw=2)
        _axes[0].set(xlabel="x at mid-height", ylabel="Concentration")
        _axes[0].legend(fontsize=10)
        _im = _axes[1].imshow(concentrations[:, :, 1].T, origin="lower", cmap="gray_r",
                              extent=(0, 1, 0, 1))
        _axes[1].contour(X, Y, concentrations[:, :, 1], levels=8, colors="#b5473a", linewidths=0.8)
        _axes[1].set(xlabel="x", ylabel="y", title="Localized source solution")
        profile_figure.colorbar(_im, ax=_axes[1], shrink=0.85)
    mo.vstack([profile_figure, mo.md(
        "Both source fields used the same diffusion operator. A direct Cholesky "
        "method would reuse its factor; the optimized solver reuses the same "
        "eigenvalue grid."
    )])
    return (profile_figure,)


@app.cell(hide_code=True)
def pattern_plot(Rectangle, np, plt):
    # Four-by-four grid illustration only: the real solve never allocates A.
    s = 4
    n = s*s
    A_small = np.zeros((n, n))
    for i in range(s):
        for j in range(s):
            k = i*s + j
            A_small[k, k] = 4
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < s and 0 <= jj < s:
                    A_small[k, ii*s + jj] = -1
    L_small = np.linalg.cholesky(A_small)

    def draw_pattern(ax, matrix, label, color):
        for row in range(n):
            for col in range(n):
                ax.add_patch(Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    facecolor=color if abs(matrix[row, col]) > 1e-12 else "white",
                    edgecolor="#8a8f98", lw=0.35,
                ))
        ax.set(xlim=(-0.5, n - 0.5), ylim=(n - 0.5, -0.5), aspect="equal",
               xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
            spine.set_color("black")
        ax.text(0.5, -0.08, label, transform=ax.transAxes,
                ha="center", va="top", color=color, fontsize=18)

    with plt.rc_context({"font.size": 14, "font.family": "Arial"}):
        pattern_figure = plt.figure(figsize=(7.0, 2.45))
        _axes = [pattern_figure.add_axes([left, 0.19, 0.265, 0.757])
                 for left in [0.01, 0.365, 0.72]]
        draw_pattern(_axes[0], A_small, r"$\mathbf{A}$", "#8a8f98")
        # L_small is the lower factor; the page names its transpose L.
        draw_pattern(_axes[1], L_small, r"$\mathbf{L}^{\mathsf{T}}$", "#b5473a")
        draw_pattern(_axes[2], L_small.T, r"$\mathbf{L}$", "#b5473a")
        pattern_figure.text(0.32, 0.57, "=", ha="center", va="center", fontsize=22)
        pattern_figure.text(0.675, 0.57, "×", ha="center", va="center", fontsize=22)
    return A_small, L_small, pattern_figure


if __name__ == "__main__":
    app.run()
