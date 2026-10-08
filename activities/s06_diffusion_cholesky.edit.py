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
    from scipy.linalg import cholesky_banded, cho_solve_banded
    return cho_solve_banded, cholesky_banded, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## S06 · Same matrix, different right-hand sides

    Take the diffusion matrix with 2 on its diagonal and −1 on the two neighbouring
    diagonals. The supplied right-hand sides describe two different source profiles
    in a layer whose face concentrations are fixed at zero.

    **Predict:** which part of the calculation can we keep when only the source changes?
    Compare the dense operation estimates below, then find the two calls that perform
    the actual **banded Cholesky** calculation.
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    grid_form = mo.ui.dropdown(
        [10, 100, 1000, 10_000, 100_000, 1_000_000], value=1000,
        label="Unknowns n",
    ).form(submit_button_label="Calculate")
    mo.vstack([grid_form, mo.md(
        "Start with 1000 unknowns. The million-unknown case is optional and uses "
        "roughly 150 MB of working arrays, plus browser overhead."
    )])
    return (grid_form,)


@app.cell
def operation_estimates(grid_form):
    N = int(grid_form.value) if grid_form.value is not None else 1000
    number_of_rhs = 10  # Change this to compare one system with many systems.
    elimination_ops = number_of_rhs*((2/3)*N**3 + 2*N**2)
    lu_ops = (2/3)*N**3 + number_of_rhs*2*N**2
    cholesky_ops = (1/3)*N**3 + number_of_rhs*2*N**2
    return N, cholesky_ops, elimination_ops, lu_ops, number_of_rhs


@app.cell(hide_code=True)
def show_estimates(N, cholesky_ops, elimination_ops, lu_ops, mo, number_of_rhs):
    mo.md(f"""
    ### Dense operation estimates: {N:,} unknowns, {number_of_rhs} right-hand sides

    | Approach | Approximate operations |
    |---|---:|
    | Fresh elimination for every system | {elimination_ops:,.0f} |
    | LU once, then reuse | {lu_ops:,.0f} |
    | Cholesky once, then reuse | {cholesky_ops:,.0f} |

    These are arithmetic estimates, not timings. They treat the matrix as **dense**.
    The actual solve below uses its narrow band and performs much less work.
    Banded LU and banded Cholesky both have linear cost for this tridiagonal matrix.
    """)
    return


@app.cell(hide_code=True)
def supplied_data(N, np):
    thickness = 1e-3
    diffusivity = 1e-10
    source_rate = 1e-4
    h = thickness/(N + 1)
    x = h*np.arange(1, N + 1)
    sources = source_rate*np.column_stack([
        np.ones(N), 1 + 0.5*np.sin(2*np.pi*x/thickness)
    ])
    rhs = (h*h/diffusivity)*sources
    return diffusivity, h, rhs, source_rate, sources, thickness, x


@app.cell(hide_code=True)
def storage_text(mo):
    mo.md(r"""
    ### The actual solve: factor once, solve two right-hand sides

    The band array `ab` stores the diagonal in row 0 and the lower neighbouring
    diagonal in row 1. Symmetry supplies the upper diagonal. The last element of
    row 1 is unused padding. The two columns of `rhs` contain the supplied sources.
    """)
    return


@app.cell
def band_storage(N, np):
    ab = np.zeros((2, N))
    ab[0, :] = 2.0
    ab[1, :-1] = -1.0
    return (ab,)


@app.cell
def factor_once(ab, cholesky_banded):
    L_band = cholesky_banded(ab, lower=True)
    return (L_band,)


@app.cell
def solve_profiles(L_band, cho_solve_banded, rhs):
    concentrations = cho_solve_banded((L_band, True), rhs)
    return (concentrations,)


@app.cell(hide_code=True)
def check_solution(concentrations, diffusivity, h, np, rhs, source_rate, sources, thickness, x):
    # Apply the three matrix diagonals directly, without allocating a dense A.
    stencil_values = 2*concentrations.copy()
    stencil_values[1:, :] -= concentrations[:-1, :]
    stencil_values[:-1, :] -= concentrations[1:, :]
    residuals = stencil_values - rhs
    scaled_residuals = np.max(np.abs(residuals), axis=0)/(
        4*np.max(np.abs(concentrations), axis=0) + np.max(np.abs(rhs), axis=0)
    )
    reference = source_rate*x*(thickness-x)/(2*diffusivity)
    reference_peak = source_rate*thickness**2/(8*diffusivity)
    relative_reference_error = np.max(np.abs(concentrations[:, 0]-reference))/reference_peak
    outward_flux = diffusivity*(concentrations[0, :] + concentrations[-1, :])/h
    discrete_generation = h*sources.sum(axis=0)
    flux_balance_error = np.abs(outward_flux-discrete_generation)/discrete_generation
    return (flux_balance_error, outward_flux, reference, reference_peak,
            relative_reference_error, residuals, scaled_residuals)


@app.cell(hide_code=True)
def results(N, ab, L_band, mo, np, residuals):
    mo.md(f"""
    **Largest absolute residual, either source:** {np.max(np.abs(residuals)):.2e} mol/m³.

    | Storage | Size |
    |---|---:|
    | Dense matrix, if allocated | {8*N*N/1e9:,.3f} GB |
    | Actual band array | {ab.nbytes/1e6:.3f} MB |
    | Reusable Cholesky factor | {L_band.nbytes/1e6:.3f} MB |

    Sizes use decimal MB/GB and exclude solutions and other working arrays.
    """)
    return


@app.cell(hide_code=True)
def profile_plot(N, concentrations, mo, np, plt, reference_peak, thickness, x):
    _indices = np.unique(np.linspace(0, N-1, min(N, 1000), dtype=int))
    _xp = np.r_[0.0, x[_indices], thickness]*1e3
    with plt.rc_context({"font.size": 13}):
        profile_figure, _ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
        for _j, _name, _color in [(0, "Uniform source", "#231f20"),
                                 (1, "Source biased to left", "#b5473a")]:
            _ax.plot(_xp, np.r_[0.0, concentrations[_indices, _j], 0.0],
                     label=_name, color=_color, lw=2)
        _ax.plot(0.5*thickness*1e3, reference_peak, "o", color="#231f20", ms=5)
        _ax.set(xlabel="Position through layer (mm)", ylabel="Concentration (mol/m³)")
        _ax.legend(fontsize=11)
    mo.vstack([profile_figure, mo.md(
        "**Discuss:** the two answers differ, but they used the same factor. "
        "Which function would you call again for a third right-hand side?"
    )])
    return (profile_figure,)


@app.cell(hide_code=True)
def pattern_plot(np, plt):
    # Eight-unknown illustration only: no large dense matrices are constructed.
    A_small = 2*np.eye(8)-np.eye(8, k=1)-np.eye(8, k=-1)
    L_small = np.linalg.cholesky(A_small)
    with plt.rc_context({"font.size": 14}):
        pattern_figure, _axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
        for _ax, _matrix, _title, _color in zip(
            _axes, [A_small, L_small], ["A: three diagonals", "L: two diagonals"],
            ["#231f20", "#b5473a"],
        ):
            _rows, _cols = np.nonzero(_matrix)
            _ax.scatter(_cols, _rows, marker="s", s=180, color=_color)
            _ax.set(xlim=(-0.5, 7.5), ylim=(7.5, -0.5), aspect="equal",
                    title=_title, xlabel="Column", ylabel="Row",
                    xticks=[0, 2, 4, 6], yticks=[0, 2, 4, 6])
    return A_small, L_small, pattern_figure


if __name__ == "__main__":
    app.run()
