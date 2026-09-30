# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "plotly>=5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


@app.cell
def playback_settings(mo):
    dimension_control = mo.ui.slider(start=6, stop=24, step=2, value=10,
                                     label="Matrix dimension n", show_value=True)
    relaxation_control = mo.ui.slider(start=0.4, stop=1.0, step=0.1, value=0.8,
                                      label="Jacobi ω (larger likely more stable)", show_value=True)
    # Display these controls beside the figure in the final cell.

    return dimension_control, relaxation_control


@app.cell
def _(dimension_control, np, relaxation_control):
    def solve_banded_demo(size, relaxation, tolerance=1e-4, seed=13, max_iterations=100):
        """Record each nonzero row elimination, back-substitution, and Jacobi sweep."""
        rng = np.random.default_rng(seed)
        # Diagonals near 10, signed band entries up to 5. Select a convergent
        # example without enforcing row-wise diagonal dominance or rescaling entries.
        for attempt in range(10000):
            A = np.zeros((size, size))
            for band in range(1, 4):
                coupling = rng.uniform(-5.0, 5.0, size=size-band)
                A += np.diag(coupling, band) + np.diag(coupling, -band)
            diagonal = rng.uniform(9.0, 11.0, size=size)
            normalized_bands = A / np.sqrt(diagonal[:, None] * diagonal[None, :])
            jacobi_radius = np.max(np.abs(np.linalg.eigvalsh(normalized_bands)))
            if jacobi_radius < 0.98:
                np.fill_diagonal(A, diagonal)
                break
        else:
            raise RuntimeError("No convergent example found; choose a different seed.")
        x_known = np.linspace(-1.0, 1.0, size)
        b = A @ x_known
        norm_b = np.linalg.norm(b)

        matrix, rhs = A.copy(), b.copy()
        solution = np.full(size, np.nan)
        direct = []
        eliminations = 0
        solved = 0

        def record(operation):
            direct.append(dict(matrix=matrix.copy(), rhs=rhs.copy(), x=solution.copy(),
                               eliminations=eliminations, solved=solved, operation=operation))

        record("Original equations")
        # Symmetric positive-definite construction gives nonzero elimination pivots.
        # This transparent example therefore needs no row swaps.
        for pivot in range(size-1):
            for row in range(pivot+1, size):
                if matrix[row, pivot] == 0:
                    continue
                multiplier = matrix[row, pivot] / matrix[pivot, pivot]
                matrix[row, pivot:] -= multiplier * matrix[pivot, pivot:]
                matrix[row, pivot] = 0.0
                rhs[row] -= multiplier * rhs[pivot]
                eliminations += 1
                record(f"R{row+1} ← R{row+1} − ({multiplier:.3f}) R{pivot+1}")
        u_step = len(direct)-1
        for row in range(size-1, -1, -1):
            solution[row] = (rhs[row] - matrix[row, row+1:] @ solution[row+1:]) / matrix[row, row]
            solved += 1
            record(f"Back-substitution: solve x{row+1}")

        diagonal = np.diag(A)
        guess = np.zeros(size)
        residual = np.linalg.norm(b-A@guess)/norm_b
        jacobi = [dict(x=guess.copy(), residual=residual)]
        for sweep in range(1, max_iterations+1):
            guess = guess + relaxation * (b-A@guess)/diagonal
            residual = np.linalg.norm(b-A@guess)/norm_b
            jacobi.append(dict(x=guess.copy(), residual=residual))
            if residual <= tolerance:
                break
        converged = residual <= tolerance
        assert np.allclose(solution, np.linalg.solve(A, b))
        assert np.allclose(solution, x_known)
        assert np.allclose(np.tril(direct[u_step]["matrix"], -1), 0)
        return dict(A=A, b=b, known=x_known, direct=direct, jacobi=jacobi,
                    u_step=u_step, converged=converged, tolerance=tolerance,
                    relaxation=relaxation)

    n = dimension_control.value
    omega = relaxation_control.value
    solver_run = solve_banded_demo(n, omega, tolerance=1.e-4)
    direct_history = solver_run["direct"]
    jacobi_history = solver_run["jacobi"]
    frame_count = max(len(direct_history), len(jacobi_history))

    return direct_history, frame_count, jacobi_history, n, omega, solver_run


@app.cell
def plotly_animation(
    dimension_control,
    direct_history,
    frame_count,
    go,
    jacobi_history,
    make_subplots,
    mo,
    n,
    np,
    omega,
    relaxation_control,
    solver_run,
):
    # Plotly can retain registered frames when a figure is reactively updated.
    # Use fresh names and an explicit playback list so old runs cannot be replayed.
    from uuid import uuid4 as _uuid4
    _run_key = _uuid4().hex
    _frame_names = [f"{_run_key}-{step}" for step in range(frame_count)]

    # A full blank column separates matrix coefficients from the right-hand side.
    _matrix_limit = max(np.max(np.abs(s["matrix"])) for s in direct_history)
    _rhs_limit = max(np.max(np.abs(s["rhs"])) for s in direct_history)
    # Matrix coefficients and RHS have separate fixed scales, shared left/right.
    _x_limit = max(np.max(np.abs(solver_run["known"])),
                   max(np.max(np.abs(s["x"])) for s in jacobi_history))
    _font_size = 11 if n <= 12 else 8


    def matrix_trace(matrix):
        return go.Heatmap(z=matrix, x=list(range(n)), y=list(range(n)),
            colorscale="RdBu_r", zmin=-_matrix_limit, zmax=_matrix_limit,
            showscale=False, xgap=1, ygap=1,
            texttemplate="%{z:.2f}" if n <= 12 else "",
            textfont=dict(size=_font_size), hovertemplate="Coefficient: %{z:.6g}<extra></extra>")


    def rhs_trace(rhs):
        # Centers n-1 and n+1 leave one full white column between A and the RHS.
        return go.Heatmap(z=rhs[:, None], x=[n+1], y=list(range(n)),
            colorscale="RdBu_r", zmin=-_rhs_limit, zmax=_rhs_limit,
            showscale=False, xgap=1, ygap=1,
            texttemplate="%{z:.2f}" if n <= 12 else "",
            textfont=dict(size=_font_size), hovertemplate="Right-hand side: %{z:.6g}<extra></extra>")


    def solution_trace(values):
        # Unknown values have a visible '?' rather than a different vector layout.
        labels = [["?" if np.isnan(v) else f"{v:.3f}" for v in values]]
        return go.Heatmap(z=np.nan_to_num(values, nan=0.0)[None, :],
            x=list(range(n)), y=[0], text=labels,
            texttemplate="%{text}", textfont=dict(size=_font_size),
            colorscale="RdBu_r", zmin=-_x_limit, zmax=_x_limit,
            showscale=False, xgap=1, ygap=1,
            hovertemplate="Component %{x}: %{text}<extra></extra>")


    def frame_annotations(step):
        direct = direct_history[min(step, len(direct_history)-1)]
        jstep = min(step, len(jacobi_history)-1)
        iteration = jacobi_history[jstep]
        reached_u = step >= solver_run["u_step"]
        is_converged = iteration["residual"] <= solver_run["tolerance"]
        if reached_u:
            left_status = f"✓ U reached after {solver_run['u_step']} row eliminations"
            left_color = "#dcfce7"
        else:
            left_status = f"Eliminating · {direct['eliminations']} row operations"
            left_color = "#eff6ff"
        if is_converged:
            right_status = f"✓ Converged after {jstep} Jacobi sweeps"
            right_color = "#dcfce7"
        elif jstep == len(jacobi_history)-1:
            right_status = f"Stopped at {jstep} sweeps · target not reached"
            right_color = "#fee2e2"
        else:
            right_status = f"Iterating · {jstep} Jacobi sweeps"
            right_color = "#eff6ff"
        left_detail = direct["operation"]
        if direct["solved"] == n:
            left_detail = f"Back-substitution complete · {n}/{n} components solved"
        elif reached_u:
            left_detail += f" · {direct['solved']}/{n} components solved"
        right_detail = (f"ω = {omega:.1f} · residual {iteration['residual']:.2e}"
                        f" · target ≤ {solver_run['tolerance']:.0e}")
        annotations = []
        for x, heading, status, detail, color in [
            (0.225, "Direct · Gaussian elimination", left_status, left_detail, left_color),
            (0.775, "Iterative · relaxed Jacobi", right_status, right_detail, right_color),
        ]:
            for y, text, size, background in [
                (1.15, f"<b>{heading}</b>", 17, "white"),
                (1.085, f"<b>{status}</b>", 13, color),
                (1.025, detail, 11, "white"),
                (0.19, "<b>Solution x</b>", 13, "white"),
            ]:
                annotations.append(dict(x=x, y=y, xref="paper", yref="paper", text=text,
                    showarrow=False, xanchor="center", yanchor="middle",
                    font=dict(size=size, color="#17212b"), bgcolor=background, borderpad=5))
        return annotations


    solver_figure = make_subplots(rows=2, cols=2, row_heights=[0.85, 0.15],
                                  vertical_spacing=0.15, horizontal_spacing=0.10)
    solver_figure.add_trace(matrix_trace(solver_run["A"]), row=1, col=1)
    solver_figure.add_trace(rhs_trace(solver_run["b"]), row=1, col=1)
    solver_figure.add_trace(matrix_trace(solver_run["A"]), row=1, col=2)
    solver_figure.add_trace(rhs_trace(solver_run["b"]), row=1, col=2)
    solver_figure.add_trace(solution_trace(direct_history[0]["x"]), row=2, col=1)
    solver_figure.add_trace(solution_trace(jacobi_history[0]["x"]), row=2, col=2)
    # Frame updates inherit axes, scales, and formatting from the initial traces.
    # Run-specific names prevent these updates from mixing with older animations.
    def solution_update(values):
        return go.Heatmap(z=np.nan_to_num(values, nan=0.0)[None, :],
            text=[["?" if np.isnan(v) else f"{v:.3f}" for v in values]])

    solver_figure.frames = [go.Frame(name=_frame_names[step], group=_run_key,
        traces=[0, 1, 4, 5], data=[
        go.Heatmap(z=direct_history[min(step, len(direct_history)-1)]["matrix"]),
        go.Heatmap(z=direct_history[min(step, len(direct_history)-1)]["rhs"][:, None]),
        solution_update(direct_history[min(step, len(direct_history)-1)]["x"]),
        solution_update(jacobi_history[min(step, len(jacobi_history)-1)]["x"]),
    ], layout=go.Layout(annotations=frame_annotations(step))) for step in range(frame_count)]

    for _col in [1, 2]:
        solver_figure.update_xaxes(row=1, col=_col, tickmode="array", tickvals=list(range(n+2)),
            ticktext=[str(i+1) for i in range(n)] + ["", "rhs"], fixedrange=True,
            range=[-0.5, n+1.5], tickfont_size=10)
        solver_figure.update_yaxes(row=1, col=_col, tickmode="array", tickvals=list(range(n)),
            ticktext=[str(i+1) for i in range(n)], range=[n-0.5, -0.5], fixedrange=True,
            tickfont_size=10)
        solver_figure.update_xaxes(row=2, col=_col, tickmode="array", tickvals=list(range(n)),
            ticktext=[f"x{i+1}" for i in range(n)], range=[-0.5, n-0.5], fixedrange=True,
            tickfont_size=10)
        solver_figure.update_yaxes(row=2, col=_col, showticklabels=False,
                                  range=[0.5, -0.5], fixedrange=True)
    solver_figure.update_layout(
        height=720, autosize=True, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=25, t=105, b=100), font=dict(size=12),
        annotations=frame_annotations(0), uirevision=_run_key, datarevision=_run_key,
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.14,
            xanchor="left", yanchor="top", showactive=False, buttons=[
                dict(label="▶ Play", method="animate", args=[_frame_names, dict(fromcurrent=True,
                    mode="immediate", frame=dict(duration=500, redraw=True), transition=dict(duration=0))]),
                dict(label="Pause", method="animate", args=[[None], dict(mode="immediate",
                    frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
                dict(label="Reset", method="animate", args=[[_frame_names[0]], dict(mode="immediate",
                    frame=dict(duration=0, redraw=True), transition=dict(duration=0))]),
            ])],
        sliders=[dict(x=0.36, len=0.64, y=-0.12, active=0,
            currentvalue=dict(prefix="Step "), steps=[dict(label=str(step), method="animate",
                args=[[_frame_names[step]], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for step in range(frame_count)])],
    )
    mo.vstack([
        # mo.Html('<div style="font-size:36px;font-weight:700;text-align:center;margin:0;">A x = b</div>'),
        mo.md("## Solving $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$: Comparison of Methods"),
        mo.hstack([dimension_control, relaxation_control], justify="start", gap=2),
        solver_figure,
    ], gap=0.25)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
