# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # `Ax=b` from a one-dimensional materials balance

    Consider a bar or layered thermal path with fixed temperatures at its
    ends. At each interior node, the steady conductive balance is

    $$
    G(T_i-T_{i-1})+G(T_i-T_{i+1})=q_i,
    $$

    where $G$ is a thermal conductance and $q_i$ is a heat source. Writing
    all node balances together produces

    $$
    A\mathbf{T}=\mathbf{b}.
    $$

    The matrix is not arbitrary: its diagonal and off-diagonal entries
    record which neighboring material nodes exchange heat. We will inspect
    that structure, implement Gaussian elimination with partial pivoting,
    and compare it with a trusted library solve.
    """)
    return


@app.cell
def _(mo):
    interior_nodes = mo.ui.slider(
        start=3,
        stop=8,
        step=1,
        value=4,
        label="Number of interior nodes",
        show_value=True,
    )
    conductance = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.5,
        value=1.5,
        label="Conductance G (W/K)",
        show_value=True,
    )
    source = mo.ui.slider(
        start=-1.0,
        stop=2.0,
        step=0.5,
        value=0.5,
        label="Heat source per interior node q (W)",
        show_value=True,
    )
    mo.vstack([interior_nodes, conductance, source], align="start", gap=1)
    return conductance, interior_nodes, source


@app.cell
def _(np):
    def assemble_balance_system(
        n_interior,
        conductance,
        source,
        left_temperature=600.0,
        right_temperature=300.0,
    ):
        matrix = np.zeros((n_interior, n_interior), dtype=float)
        rhs = np.full(n_interior, source, dtype=float)
        for i in range(n_interior):
            matrix[i, i] += 2.0 * conductance
            if i == 0:
                rhs[i] += conductance * left_temperature
            else:
                matrix[i, i - 1] -= conductance
            if i == n_interior - 1:
                rhs[i] += conductance * right_temperature
            else:
                matrix[i, i + 1] -= conductance
        return matrix, rhs

    return (assemble_balance_system,)


@app.cell
def _(np):
    def gaussian_elimination(matrix, rhs):
        # Given algorithm: inspect the pivot, elimination, and back-substitution.
        a = matrix.copy().astype(float)
        b = rhs.copy().astype(float)
        n = len(b)

        for pivot_column in range(n - 1):
            pivot_row = pivot_column + abs(
                a[pivot_column:, pivot_column]
            ).argmax()
            if abs(a[pivot_row, pivot_column]) < 1.0e-14:
                raise ValueError("singular or nearly singular matrix")
            if pivot_row != pivot_column:
                a[[pivot_column, pivot_row]] = a[[pivot_row, pivot_column]]
                b[[pivot_column, pivot_row]] = b[[pivot_row, pivot_column]]

            for row in range(pivot_column + 1, n):
                factor = a[row, pivot_column] / a[pivot_column, pivot_column]
                a[row, pivot_column:] -= factor * a[pivot_column, pivot_column:]
                b[row] -= factor * b[pivot_column]

        solution = np.zeros(n, dtype=float)
        for row in range(n - 1, -1, -1):
            known = np.dot(a[row, row + 1 :], solution[row + 1 :])
            solution[row] = (b[row] - known) / a[row, row]
        return solution

    return (gaussian_elimination,)


@app.cell
def _(
    assemble_balance_system,
    conductance,
    gaussian_elimination,
    interior_nodes,
    np,
    source,
):
    matrix, rhs = assemble_balance_system(
        int(interior_nodes.value), conductance.value, source.value
    )
    manual_solution = gaussian_elimination(matrix, rhs)
    library_solution = np.linalg.solve(matrix, rhs)
    manual_residual = matrix @ manual_solution - rhs
    library_residual = matrix @ library_solution - rhs
    return library_solution, manual_residual, manual_solution, matrix, rhs


@app.cell
def _(library_solution, manual_residual, manual_solution, matrix, mo, np, rhs):
    matrix_text = np.array2string(matrix, precision=2, suppress_small=True)
    rhs_text = np.array2string(rhs, precision=2, suppress_small=True)
    difference = np.max(np.abs(manual_solution - library_solution))
    mo.callout(
        mo.md(
            f"""
            ## Linear-system result

            ```text
            A =
            {matrix_text}

            b = {rhs_text}
            ```

            - Gaussian-elimination temperatures (K):
              **{np.array2string(manual_solution, precision=3)}**
            - `numpy.linalg.solve` temperatures (K):
              **{np.array2string(library_solution, precision=3)}**
            - maximum difference between implementations: **{difference:.3e} K**
            - Gaussian-elimination residual norm:
              **{np.linalg.norm(manual_residual, ord=np.inf):.3e} W**

            The matrix encodes local neighbor coupling. Agreement with an
            independent implementation is a verification check, not proof that
            the thermal model or its conductance is appropriate.
            """
        ),
        kind="success" if difference < 1.0e-10 else "warn",
    )
    return


@app.cell
def _(library_solution, mo, np):
    temperatures = np.concatenate(([600.0], library_solution, [300.0]))
    x_values = np.arange(len(temperatures), dtype=float)
    x_pixels = 60 + x_values / max(x_values[-1], 1.0) * 600
    t_min, t_max = temperatures.min(), temperatures.max()
    y_pixels = 30 + (t_max - temperatures) / max(t_max - t_min, 1.0) * 220
    points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in zip(x_pixels, y_pixels)
    )
    mo.Html(
        f"""
        <figure aria-label="Temperature profile through the material bar">
          <svg viewBox="0 0 700 285" role="img"
               style="max-width: 700px; width: 100%; height: auto;">
            <title>Steady temperature decreases along the bar</title>
            <polyline points="{points}" fill="none" stroke="#007c41"
                      stroke-width="3" />
            <line x1="60" y1="250" x2="660" y2="250" stroke="currentColor" />
            <line x1="60" y1="30" x2="60" y2="250" stroke="currentColor" />
            <text x="360" y="278" text-anchor="middle" font-size="14">node position</text>
            <text x="18" y="140" text-anchor="middle" font-size="14"
                  transform="rotate(-90 18 140)">temperature (K)</text>
          </svg>
        </figure>
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### What to inspect

            Open **Edit code** and inspect `gaussian_elimination`. The important
            parts are not Python syntax alone: pivoting protects the calculation,
            elimination creates an upper-triangular system, and back-substitution
            recovers the unknown temperatures. Compare each step with the matrix
            structure and the physical balance at a node.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
