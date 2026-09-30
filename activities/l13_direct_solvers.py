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
    from fractions import Fraction
    from uuid import uuid4

    return Fraction, go, mo, np, uuid4


@app.cell
def controls(mo):

    dimension = mo.ui.number(start=2, stop=6, step=1, value=3, label="Matrix order n")
    random_button = mo.ui.button(
        value=0, label="🎲 Random A, b", on_click=lambda count: count + 1,
    )
    method = mo.ui.dropdown(
        options=["Gaussian elimination", "Gauss–Jordan elimination", "LU factorization"],
        value="Gaussian elimination",
        label="Method",
    )
    pivoting = mo.ui.checkbox(value=True, label="Partial pivoting")
    a11_choices = {"as in A": None, "0": 0.0, "10⁻⁴": 1e-4, "10⁻⁸": 1e-8,
                   "10⁻¹²": 1e-12, "10⁻¹⁴": 1e-14, "10⁻¹⁶": 1e-16}
    small_pivot = mo.ui.dropdown(options=list(a11_choices), value="as in A", label="Set a₁₁ to")
    mo.vstack([
        mo.hstack([dimension, random_button, method], justify="start", align="center", gap=1.5),
        mo.hstack([pivoting, small_pivot], justify="start", align="center", gap=1.5),
    ], gap=0.4)

    return a11_choices, dimension, method, pivoting, random_button, small_pivot


@app.cell
def matrix_inputs(dimension, mo, np, random_button):

    n_input = int(dimension.value)
    random_button.value  # a click draws a new system

    # Half-integer entries and an integer solution keep the arithmetic readable.
    _rng = np.random.default_rng()
    for _attempt in range(200):
        A_initial = _rng.integers(-10, 11, size=(n_input, n_input)) / 2
        if np.linalg.cond(A_initial) < 30:
            break
    _x_initial = _rng.integers(-3, 4, size=n_input).astype(float)
    b_initial = A_initial @ _x_initial
    _labels = [str(i + 1) for i in range(n_input)]

    A_editor = mo.ui.matrix(
        A_initial, min_value=-10, max_value=10, step=0.5, precision=1,
        row_labels=_labels, column_labels=_labels, debounce=True, label="A",
    )
    b_editor = mo.ui.matrix(
        b_initial.reshape(-1, 1), min_value=-60, max_value=60, step=0.5, precision=1,
        row_labels=_labels, column_labels=["b"], debounce=True, label="b",
    )
    mo.vstack([
        mo.hstack([A_editor, b_editor], justify="start", align="start", gap=1),
        mo.md("<small>Drag an entry sideways to change it.</small>"),
    ], gap=0.4)

    return A_editor, b_editor


@app.cell
def current_system(A_editor, a11_choices, b_editor, np, small_pivot):

    A = np.asarray(A_editor.value, dtype=float)
    b = np.asarray(b_editor.value, dtype=float).reshape(-1)
    if a11_choices[small_pivot.value] is not None:
        A[0, 0] = a11_choices[small_pivot.value]

    return A, b


@app.cell
def direct_solver(A, Fraction, b, method, np, pivoting):

    def solve_direct(A, b, method, partial_pivoting=True):
        """Record each solver step; return (states, solution or None, pivots used)."""
        n = len(b)
        if A.shape != (n, n):
            raise ValueError("A must be square and its order must match the length of b.")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("Enter finite numbers in every matrix and vector entry.")
        # With pivoting, a candidate at round-off level counts as zero. Without pivoting,
        # only an exact zero stops elimination, so tiny pivots are used as they are.
        threshold = np.finfo(float).eps * max(1.0, np.linalg.norm(A, ord=np.inf)) * n * 10
        states, pivots = [], []

        def record(blocks, phase, detail, op=None):
            states.append({
                "blocks": {name: np.array(value, dtype=float, copy=True) for name, value in blocks.items()},
                "phase": phase, "detail": detail, "op": op,
            })

        def choose_pivot(column, col, blocks, block_name):
            """Return the pivot row, or None after recording a zero-pivot step."""
            if partial_pivoting:
                pivot = col + int(np.argmax(np.abs(column[col:])))
                if abs(column[pivot]) > threshold:
                    return pivot
                candidates = f"rows {col + 1}–{n}" if col < n - 1 else f"row {n}"
                detail = (f"column {col + 1} is zero (to round-off) in {candidates}, so no row swap can supply "
                          "a pivot: A is singular")
                rows = (col, n - 1)
            else:
                if column[col] != 0:
                    return col
                detail = (f"the pivot in row {col + 1} is zero and pivoting is off, "
                          "so elimination stops here")
                rows = (col, col)
            record(blocks, "Cannot solve!", detail,
                   {"kind": "zero", "block": block_name, "col": col, "rows": rows})
            return None

        if method in ("Gaussian elimination", "Gauss–Jordan elimination"):
            M = np.column_stack((A, b)).astype(float)
            x = np.full(n, np.nan)

            def blocks():
                return {"A": M[:, :n], "b": M[:, n], "x": x}

            record(blocks(), "Start", "augmented matrix [A | b]")
            jordan = method == "Gauss–Jordan elimination"
            for col in range(n):
                pivot = choose_pivot(M[:, col], col, blocks(), "A")
                if pivot is None:
                    return states, None, pivots
                pivots.append(M[pivot, col])
                if pivot != col:
                    M[[col, pivot]] = M[[pivot, col]]
                    record(blocks(), "Row swap", f"R{col + 1} ↔ R{pivot + 1} puts the largest |pivot| on the diagonal",
                           {"kind": "swap", "rows": (col, pivot)})
                if jordan:
                    pivot_value = M[col, col]
                    M[col] /= pivot_value
                    record(blocks(), "Scale pivot row", f"R{col + 1} ← R{col + 1} ÷ {fmt(pivot_value)}",
                           {"kind": "scale", "row": col, "divisor": pivot_value})
                rows = [r for r in range(n) if r != col] if jordan else range(col + 1, n)
                for row in rows:
                    if M[row, col] == 0:
                        continue
                    factor = M[row, col] / M[col, col]
                    M[row] -= factor * M[col]
                    M[row, col] = 0.0
                    record(blocks(), "Elimination",
                           f"R{row + 1} ← R{row + 1} + ({fmt(-factor)}) × R{col + 1}",
                           {"kind": "add", "source": col, "target": row, "factor": -factor, "block": "A"})
            if jordan:
                x[:] = M[:, n]
                record(blocks(), "Identity reached", "[I | x]: the last column is the solution",
                       {"kind": "done"})
                return states, x.copy(), pivots
            record(blocks(), "Upper triangular", "[U | c] reached; back-substitute from the last row")
            for row in range(n - 1, -1, -1):
                x[row] = (M[row, n] - M[row, row + 1:n] @ x[row + 1:]) / M[row, row]
                record(blocks(), "Back-substitution",
                       f"x{row + 1} = (c{row + 1} − Σ u{row + 1}j xj) ÷ u{row + 1}{row + 1} = {fmt(x[row])}",
                       {"kind": "solve", "row": row, "matrix": "A", "rhs": "b", "result": "x"})
            return states, x.copy(), pivots

        # LU: start from A = I·A and move each elimination multiplier into L.
        L = np.eye(n)
        U = A.astype(float).copy()
        Pb = b.astype(float).copy()
        y = np.full(n, np.nan)
        x = np.full(n, np.nan)

        def blocks():
            return {"L": L, "Pb": Pb, "y": y, "U": U, "x": x}

        record(blocks(), "Start", "A = I·A: L = I and U = A")
        for col in range(n):
            pivot = choose_pivot(U[:, col], col, blocks(), "U")
            if pivot is None:
                return states, None, pivots
            pivots.append(U[pivot, col])
            if pivot != col:
                U[[col, pivot]] = U[[pivot, col]]
                L[[col, pivot], :col] = L[[pivot, col], :col]
                Pb[[col, pivot]] = Pb[[pivot, col]]
                record(blocks(), "Row swap",
                       f"R{col + 1} ↔ R{pivot + 1} in U, in the stored part of L, and in Pb",
                       {"kind": "swap", "rows": (col, pivot)})
            for row in range(col + 1, n):
                factor = U[row, col] / U[col, col]
                L[row, col] = factor
                U[row, col:] -= factor * U[col, col:]
                U[row, col] = 0.0
                record(blocks(), "Elimination",
                       f"R{row + 1} ← R{row + 1} + ({fmt(-factor)}) × R{col + 1} in U; record ℓ{row + 1}{col + 1} = {fmt(factor)} in L",
                       {"kind": "add", "source": col, "target": row, "factor": -factor, "block": "U",
                        "stored": (row, col)})
        record(blocks(), "Factorization complete",
               ("P A = L U" if partial_pivoting else "A = L U") + "; next solve L y = " + ("P b" if partial_pivoting else "b"))
        for row in range(n):
            y[row] = Pb[row] - L[row, :row] @ y[:row]
            record(blocks(), "Forward substitution",
                   f"y{row + 1} = (Pb){row + 1} − Σ ℓ{row + 1}j yj = {fmt(y[row])}",
                   {"kind": "solve", "row": row, "matrix": "L", "rhs": "Pb", "result": "y"})
        for row in range(n - 1, -1, -1):
            x[row] = (y[row] - U[row, row + 1:] @ x[row + 1:]) / U[row, row]
            record(blocks(), "Back substitution",
                   f"x{row + 1} = (y{row + 1} − Σ u{row + 1}j xj) ÷ u{row + 1}{row + 1} = {fmt(x[row])}",
                   {"kind": "solve", "row": row, "matrix": "U", "rhs": "y", "result": "x"})
        return states, x.copy(), pivots


    def exact_solution(A, b):
        """Solve with exact rational arithmetic; None when A is exactly singular."""
        n = len(b)
        M = [[Fraction(float(v)) for v in row] + [Fraction(float(rhs))] for row, rhs in zip(A, b)]
        for col in range(n):
            pivot = next((r for r in range(col, n) if M[r][col] != 0), None)
            if pivot is None:
                return None
            M[col], M[pivot] = M[pivot], M[col]
            for row in range(n):
                if row != col and M[row][col] != 0:
                    factor = M[row][col] / M[col][col]
                    M[row] = [a - factor * p for a, p in zip(M[row], M[col])]
        return np.array([float(M[i][n] / M[i][i]) for i in range(n)])


    def fmt(value):
        """Short label: three decimals, or powers of ten for very small or large values."""
        if not np.isfinite(value):
            return "?"
        if value != 0 and (abs(value) < 5e-4 or abs(value) >= 1e5):
            mantissa, exponent = f"{value:.0e}".split("e")
            return f"{mantissa}e{int(exponent)}"
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text


    try:
        solver_states, solution, solver_pivots = solve_direct(A, b, method.value, pivoting.value)
        solver_error = None
    except ValueError as error:
        solver_states, solution, solver_pivots = [], None, []
        solver_error = str(error)
    x_exact = exact_solution(A, b) if solver_error is None else None

    return fmt, solver_error, solver_states, x_exact


@app.cell
def animation(
    fmt,
    go,
    method,
    mo,
    np,
    solver_error,
    solver_states,
    uuid4,
    x_exact,
):

    def block_layout(n, method_name):
        """Column position and header for each displayed block."""
        if method_name == "LU factorization":
            specs = [("L", "L", n, 0.0), ("Pb", "P b", 1, n + 0.3), ("y", "y", 1, n + 1.4),
                     ("U", "U", n, n + 2.9), ("x", "x", 1, 2 * n + 3.2)]
        else:
            specs = [("A", "A", n, 0.0), ("b", "b", 1, n + 0.3), ("x", "x", 1, n + 1.8)]
        return {name: {"header": header, "cols": cols, "x0": x0} for name, header, cols, x0 in specs}


    def build_figure(states, method_name, x_reference=None):
        n = len(states[0]["blocks"]["x"])
        layout = block_layout(n, method_name)
        names = list(layout)
        frame_names = [f"{uuid4().hex}-{i}" for i in range(len(states))]
        # One colour scale for every block, set by the starting system and the exact x.
        # Entries that grow beyond it saturate, which makes element growth easy to spot.
        values = [np.ravel(states[0]["blocks"][k]) for k in names]
        if x_reference is not None:
            values.append(x_reference)
        zlimit = max(1.0, float(np.nanmax(np.abs(np.concatenate(values)))))
        cell_px = 44
        font_size = 12 if n <= 4 else 10
        arrow_x, label_x, row_label_x = -1.05, -1.35, -0.6
        x_left = -3.3
        x_right = max(spec["x0"] + spec["cols"] for spec in layout.values()) - 0.5

        def as_grid(name, array):
            return array.reshape(n, -1) if array.ndim == 1 else array

        def trace_values(state, name):
            grid = as_grid(name, state["blocks"][name])
            text = [[fmt(v) if np.isfinite(v) else "?" for v in row] for row in grid]
            return np.nan_to_num(grid, nan=0.0), text

        def heatmap(state, name):
            z, text = trace_values(state, name)
            spec = layout[name]
            return go.Heatmap(
                z=z, text=text, x=spec["x0"] + np.arange(spec["cols"]), y=np.arange(n),
                colorscale="RdBu_r", zmin=-zlimit, zmax=zlimit, showscale=False,
                xgap=2, ygap=2, texttemplate="%{text}", textfont=dict(size=font_size),
                hovertemplate=f"{spec['header']}" + "[%{y}]: %{text}<extra></extra>",
            )

        def box(name, rows, cols=None, color="#222", dash="solid"):
            spec = layout[name]
            c0, c1 = (0, spec["cols"] - 1) if cols is None else cols
            return dict(
                type="rect", visible=True, xref="x", yref="y", line=dict(color=color, width=3, dash=dash),
                x0=spec["x0"] + c0 - 0.5, x1=spec["x0"] + c1 + 0.5,
                y0=rows[0] - 0.5, y1=rows[1] + 0.5,
            )

        def text(x, y, content, **kwargs):
            return dict(x=x, y=y, xref="x", yref="y", text=content, showarrow=False, visible=True, **kwargs)

        def arrow(tail, head, color, both=False):
            return dict(
                x=arrow_x, y=head, ax=arrow_x, ay=tail, xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, visible=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5,
                arrowcolor=color, arrowside="end+start" if both else "end",
                standoff=6, startstandoff=6,
            )

        static_annotations = [
            text(spec["x0"] + (spec["cols"] - 1) / 2, -0.95, f"<b>{spec['header']}</b>", font=dict(size=15))
            for spec in layout.values()
        ] + [
            text(row_label_x, row, f"R{row + 1}", xanchor="right", font=dict(size=12, color="#555"))
            for row in range(n)
        ]
        row_blocks = [name for name in names if name not in ("x", "y")]
        max_notes, max_shapes = 2, 2 * len(row_blocks) + 1

        def decorations(state):
            op = state["op"] or {}
            kind = op.get("kind")
            notes, shapes = [], []
            if kind == "swap":
                i, j = op["rows"]
                notes += [arrow(i, j, "#7b3294", both=True),
                          text(label_x, (i + j) / 2, "swap", xanchor="right", font=dict(size=13, color="#7b3294"))]
                shapes += [box(name, (r, r), color="#7b3294") for name in row_blocks for r in (i, j)]
            elif kind == "add":
                source, target = op["source"], op["target"]
                notes += [arrow(source, target, "#e66101"),
                          text(label_x, (source + target) / 2, f"× ({fmt(op['factor'])})",
                               xanchor="right", font=dict(size=13, color="#e66101"))]
                targets = [op["block"]] + (["b"] if op["block"] == "A" else [])
                shapes += [box(name, (source, source), color="#1a9641", dash="dash") for name in targets]
                shapes += [box(name, (target, target), color="#e66101") for name in targets]
                if "stored" in op:
                    r, c = op["stored"]
                    shapes.append(box("L", (r, r), (c, c), color="#e66101"))
            elif kind == "scale":
                row = op["row"]
                notes.append(text(label_x, row, f"÷ ({fmt(op['divisor'])})", xanchor="right",
                                  font=dict(size=13, color="#e66101")))
                shapes += [box(name, (row, row), color="#e66101") for name in ("A", "b")]
            elif kind == "solve":
                row = op["row"]
                notes.append(text(label_x, row, f"solve {op['result']}{row + 1}", xanchor="right",
                                  font=dict(size=13, color="#1a9641")))
                shapes += [box(op["matrix"], (row, row), color="#1a9641"),
                           box(op["rhs"], (row, row), color="#1a9641"),
                           box(op["result"], (row, row), color="#e66101")]
            elif kind == "zero":
                row0, row1 = op["rows"]
                notes.append(text(label_x, (row0 + row1) / 2, "cannot solve!", xanchor="right",
                                  font=dict(size=13, color="#d7191c")))
                shapes.append(box(op["block"], (row0, row1), (op["col"], op["col"]), color="#d7191c"))
            elif kind == "done":
                shapes.append(box("x", (0, n - 1), color="#1a9641"))
            # Plotly merges frame annotations and shapes by position, so every frame
            # carries the same number of each; unused slots are hidden placeholders.
            notes += [dict(text="", showarrow=False, visible=False)] * (max_notes - len(notes))
            shapes += [dict(type="rect", visible=False, x0=0, x1=0, y0=0, y1=0)] * (max_shapes - len(shapes))
            return static_annotations + notes, shapes

        def title(index):
            state = states[index]
            detail = state["detail"]
            return f"<b>Step {index}: {state['phase']}</b><br><span style='font-size:13px'>{detail}</span>"

        figure = go.Figure([heatmap(states[0], name) for name in names])
        frames = []
        for index, state in enumerate(states):
            notes, shapes = decorations(state)
            data = []
            for name in names:
                z, labels = trace_values(state, name)
                data.append(go.Heatmap(z=z, text=labels))
            frames.append(go.Frame(
                name=frame_names[index], data=data, traces=list(range(len(names))),
                layout=go.Layout(title_text=title(index), annotations=notes, shapes=shapes),
            ))
        figure.frames = frames
        notes, shapes = decorations(states[0])
        width = max(int(cell_px * (x_right - x_left)) + 20, 480)
        height = int(cell_px * (n + 0.9)) + 200
        animate = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
        figure.update_xaxes(range=[x_left, x_right], visible=False, fixedrange=True)
        figure.update_yaxes(range=[n - 0.5, -1.4], visible=False, fixedrange=True,
                            scaleanchor="x", scaleratio=1)
        figure.update_layout(
            width=width, height=height, autosize=False,
            margin=dict(l=10, r=10, t=60, b=140),
            plot_bgcolor="white", paper_bgcolor="white",
            title=dict(text=title(0), x=0.02, xanchor="left", yref="container", y=1 - 12 / height,
                       yanchor="top", font=dict(size=15)),
            annotations=notes, shapes=shapes,
            updatemenus=[dict(
                type="buttons", direction="left", x=0, xanchor="left", y=0, yanchor="top",
                pad=dict(t=45), showactive=False, buttons=[
                    dict(label="▶", method="animate", args=[frame_names, dict(
                        animate, fromcurrent=True, frame=dict(duration=900, redraw=True))]),
                    dict(label="❚❚", method="animate", args=[[None], dict(
                        animate, frame=dict(duration=0, redraw=False))]),
                ],
            )],
            sliders=[dict(
                x=110 / width, len=1 - 110 / width, y=0, yanchor="top", pad=dict(t=30),
                currentvalue=dict(prefix="Step ", font=dict(size=12)),
                steps=[dict(label=str(i), method="animate", args=[[frame_names[i]], animate])
                       for i in range(len(states))],
            )],
        )
        return figure


    if solver_error:
        output = mo.callout(mo.md(f"**Cannot solve this system:** {solver_error}"), kind="danger")
    else:
        output = build_figure(solver_states, method.value, x_exact)
    output

    return


if __name__ == "__main__":
    app.run()
