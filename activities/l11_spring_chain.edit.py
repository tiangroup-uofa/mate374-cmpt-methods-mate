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
    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L11 · A chain of atoms held together by springs

    Picture $N$ atoms sitting on a line at their rest spacing. Near that spacing,
    the energy of each bond grows roughly as the square of its stretch, so we can
    replace every bond by a spring with stiffness $k_i$ (the curvature of the
    pair potential at its minimum). When we push, pull, or clamp some of the
    atoms, the chain settles into a new static arrangement. We describe that
    arrangement by the **displacement** $u_i$ of each atom from its rest
    position, measured in Å and positive to the right.

    Spring $i$ joins atom $i$ to atom $i+1$. It stretches by $u_{i+1}-u_i$ and,
    by Hooke's law, carries the tension

    $$T_i = k_i\,(u_{i+1}-u_i).$$

    A positive $T_i$ means the spring is stretched and pulls its two atoms
    together. A negative $T_i$ means it is compressed and pushes them apart.
    In static equilibrium the net force on every atom is zero. The atomic
    masses therefore drop out of this problem. They return when the atoms
    vibrate, which leads to an eigenvalue problem later in the course.

    Set up a chain below and follow how each atom contributes one linear
    equation to a system $\mathbf K\mathbf u=\mathbf f$.
    """)
    return


@app.cell(hide_code=True)
def chain_controls(mo):
    n_atoms = mo.ui.slider(3, 8, value=4, step=1, label="Number of atoms $N$", show_value=True)
    scenario = mo.ui.dropdown(
        [
            "Pull the right end, left end clamped",
            "Stretch between two clamped ends",
            "Push a middle atom, both ends clamped",
            "Pull the right end, no atom clamped",
        ],
        value="Pull the right end, left end clamped",
        label="Starting scenario",
    )
    mo.vstack(
        [
            mo.hstack([n_atoms, scenario], justify="start", gap=2, wrap=True),
            mo.md(
                "The scenario only fills in starting values. Edit any atom or "
                "spring afterwards to build your own case."
            ),
        ]
    )
    return n_atoms, scenario


@app.cell(hide_code=True)
def atom_controls(mo, n_atoms, scenario):
    _n = n_atoms.value
    _clamped = [False] * _n
    _force = [0.0] * _n
    _disp = [0.0] * _n
    if scenario.value.startswith("Pull the right end, left"):
        _clamped[0] = True
        _force[-1] = 0.5
    elif scenario.value.startswith("Stretch"):
        _clamped[0] = _clamped[-1] = True
        _disp[-1] = 0.3
    elif scenario.value.startswith("Push"):
        _clamped[0] = _clamped[-1] = True
        _force[_n // 2] = 0.5
    else:
        _force[-1] = 0.5

    FREE = "Force applied"
    CLAMPED = "Displacement set"
    atom_status = mo.ui.array(
        [mo.ui.dropdown([FREE, CLAMPED], value=CLAMPED if _c else FREE) for _c in _clamped]
    )
    applied_force = mo.ui.array(
        [mo.ui.number(-2.0, 2.0, step=0.1, value=_f) for _f in _force]
    )
    set_displacement = mo.ui.array(
        [mo.ui.number(-0.5, 0.5, step=0.05, value=_u) for _u in _disp]
    )
    spring_k = mo.ui.array(
        [mo.ui.number(0.5, 50.0, step=0.5, value=5.0) for _ in range(_n - 1)]
    )
    return CLAMPED, FREE, applied_force, atom_status, set_displacement, spring_k


@app.cell(hide_code=True)
def control_table(applied_force, atom_status, mo, n_atoms, set_displacement, spring_k):
    _rows = [
        mo.hstack(
            [mo.md("**Atom**"), mo.md("**Condition**"), mo.md("**Force $F_i$ (eV/Å)**"),
             mo.md("**Displacement $u_i$ (Å)**")],
            widths=[1, 2, 2, 2],
        )
    ]
    for _i in range(n_atoms.value):
        _rows.append(
            mo.hstack(
                [mo.md(f"Atom {_i + 1}"), atom_status[_i], applied_force[_i], set_displacement[_i]],
                widths=[1, 2, 2, 2],
                align="center",
            )
        )
    _springs = mo.hstack(
        [mo.vstack([mo.md(f"$k_{{{_i + 1}}}$"), spring_k[_i]], align="center")
         for _i in range(n_atoms.value - 1)],
        justify="start",
        wrap=True,
    )
    mo.vstack(
        [
            mo.md(
                "### Atoms and bonds\n\n"
                "For each atom, choose which quantity you control. A free atom feels "
                "the force you apply, and its displacement is found by the solver. "
                "A clamped atom is held at the displacement you set, and the solver "
                "finds the force the clamp must supply. The unused column of each "
                "row is ignored."
            ),
            *_rows,
            mo.md("**Spring stiffnesses (eV/Å²)**, from left to right:"),
            _springs,
        ]
    )
    return


@app.cell
def chain_model(CLAMPED, applied_force, atom_status, np, set_displacement, spring_k):
    n = len(atom_status.value)
    k = np.array(spring_k.value, dtype=float)
    clamped = np.array([_s == CLAMPED for _s in atom_status.value])
    free = ~clamped
    F_applied = np.where(clamped, 0.0, np.array(applied_force.value, dtype=float))
    u_set = np.array(set_displacement.value, dtype=float)

    # Each spring i couples atoms i and i+1 and adds a 2x2 block to K.
    K = np.zeros((n, n))
    for _i in range(n - 1):
        K[_i : _i + 2, _i : _i + 2] += k[_i] * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return F_applied, K, clamped, free, k, n, u_set


@app.cell
def reduced_system(F_applied, K, clamped, free, np, u_set):
    # Keep the rows of free atoms. Known displacements move to the right-hand side.
    K_ff = K[np.ix_(free, free)]
    K_fc = K[np.ix_(free, clamped)]
    b = F_applied[free] - K_fc @ u_set[clamped]
    n_free = int(free.sum())
    rank = int(np.linalg.matrix_rank(K_ff)) if n_free else 0
    solvable = rank == n_free
    return K_fc, K_ff, b, n_free, rank, solvable


@app.cell(hide_code=True)
def unknowns_table(F_applied, clamped, mo, n, u_set):
    _lines = [
        "| Atom | Condition | Known quantity | Unknown quantity |",
        "|---:|---|---|---|",
    ]
    for _i in range(n):
        _j = _i + 1
        if clamped[_i]:
            _lines.append(
                f"| {_j} | clamped | $u_{{{_j}}} = {u_set[_i]:g}$ Å "
                f"| clamp force $R_{{{_j}}}$ |"
            )
        else:
            _lines.append(
                f"| {_j} | free | $F_{{{_j}}} = {F_applied[_i]:g}$ eV/Å "
                f"| displacement $u_{{{_j}}}$ |"
            )
    mo.md(
        "### What are the unknowns?\n\n"
        "Every atom has one displacement and one external force acting on it. "
        "Specifying one of the pair for each atom leaves exactly one unknown per "
        f"atom, so this chain has **{n} unknowns** and, as the next step shows, "
        f"**{n} force-balance equations**.\n\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def balance_equations(F_applied, K, clamped, k, mo, n, u_set):
    def _num(value):
        return f"{value:g}"

    def _row_numeric(row):
        _terms = []
        for _j, _c in enumerate(row):
            if _c == 0:
                continue
            _sign = "-" if _c < 0 else "+"
            _mag = "" if abs(_c) == 1 else _num(abs(_c))
            _terms.append(f"{_sign} {_mag}u_{{{_j + 1}}}")
        _text = " ".join(_terms)
        return _text[2:] if _text.startswith("+ ") else _text

    _eqs = []
    for _i in range(n):
        _j = _i + 1
        _parts = []
        if _i > 0:
            _parts.append(f"-k_{{{_j - 1}}}(u_{{{_j}}}-u_{{{_j - 1}}})")
        if _i < n - 1:
            _parts.append(f"+k_{{{_j}}}(u_{{{_j + 1}}}-u_{{{_j}}})")
        _symbolic = " ".join(_parts).lstrip("+")
        _external = f"R_{{{_j}}}" if clamped[_i] else f"F_{{{_j}}}"
        _rhs = f"R_{{{_j}}}" if clamped[_i] else _num(F_applied[_i])
        _eqs.append(
            rf"\text{{Atom {_j}:}}\quad & {_symbolic} + {_external} = 0"
            rf"\quad\Longrightarrow\quad {_row_numeric(K[_i])} = {_rhs}"
        )
    _k_values = ", ".join(f"k_{{{_i + 1}}}={_num(k[_i])}" for _i in range(n - 1))
    mo.md(
        "### One force balance per atom\n\n"
        "Atom $i$ is pulled left by the spring on its left, pulled right by the "
        "spring on its right, and pushed by its external force (the applied force "
        "$F_i$ or the clamp force $R_i$). Setting the sum to zero and collecting "
        "the displacements gives one linear equation per atom. The left side uses "
        "the tension formula $T_i=k_i(u_{i+1}-u_i)$, and the right side moves the "
        f"external force across the equals sign. Here ${_k_values}$ eV/Å².\n\n"
        "$$\n\\begin{aligned}\n" + r"\\[4pt]".join(_eqs) + "\n\\end{aligned}\n$$"
    )
    return


@app.cell(hide_code=True)
def matrix_form(F_applied, K, K_fc, K_ff, b, clamped, free, mo, n, np, u_set):
    _unknown = r"\color{#d95f02}"

    def _bmatrix(rows):
        return (
            r"\begin{bmatrix}"
            + r"\\".join(" & ".join(rows_i) for rows_i in rows)
            + r"\end{bmatrix}"
        )

    def _fmt(value):
        return f"{value:g}"

    _u_full = [
        [_fmt(u_set[_i])] if clamped[_i] else [f"{_unknown}u_{{{_i + 1}}}"] for _i in range(n)
    ]
    _f_full = [
        [f"{_unknown}R_{{{_i + 1}}}"] if clamped[_i] else [_fmt(F_applied[_i])] for _i in range(n)
    ]
    _K_full = [[_fmt(_c) for _c in _row] for _row in K]

    _free_ids = np.flatnonzero(free) + 1
    _clamped_ids = np.flatnonzero(clamped) + 1
    if len(_free_ids) and not len(_clamped_ids):
        _reduced_text = (
            "No atom is clamped, so every displacement is unknown and the full "
            "system above is the one `np.linalg.solve` must handle."
        )
    elif len(_free_ids):
        _u_free = [[f"{_unknown}u_{{{_j}}}"] for _j in _free_ids]
        _K_reduced = [[_fmt(_c) for _c in _row] for _row in K_ff]
        _rhs_parts = [[_fmt(_v)] for _v in F_applied[free]]
        _moved = [[_fmt(_v)] for _v in (K_fc @ u_set[clamped])]
        _reduced = (
            f"{_bmatrix(_K_reduced)}{_bmatrix(_u_free)} = {_bmatrix(_rhs_parts)}"
            f" - {_bmatrix(_moved)} = {_bmatrix([[_fmt(_v)] for _v in b])}"
        )
        _reduced_text = (
            "The clamped displacements are already known, so their columns can be "
            "multiplied out and moved to the right-hand side. The rows of the "
            "clamped atoms contain the unknown clamp forces, so we set them aside "
            f"for now. What remains is a {len(_free_ids)} × {len(_free_ids)} system "
            "for the free displacements alone:\n\n"
            f"$$\n\\mathbf K_{{\\text{{free}}}}\\,\\mathbf u_{{\\text{{free}}}}"
            f"=\\mathbf F_{{\\text{{free}}}}-\\mathbf K_{{\\text{{free,clamped}}}}\\,"
            f"\\mathbf u_{{\\text{{clamped}}}}\n$$\n\n$${_reduced}$$"
        )
    else:
        _reduced_text = (
            "Every atom is clamped, so every displacement is already known and "
            "there is no system left to solve. The balance equations only report "
            "the clamp forces."
        )

    mo.md(
        "### The same equations as a matrix\n\n"
        "Writing the coefficients of the balance equations row by row gives the "
        "**stiffness matrix** $\\mathbf K$. Row $i$ is the balance of atom $i$, "
        "and column $j$ collects every term that multiplies $u_j$. Because each "
        "spring only touches its two neighbours, $\\mathbf K$ is tridiagonal. "
        "Unknowns are shown in orange.\n\n"
        f"$$\\underbrace{{{_bmatrix(_K_full)}}}_{{\\mathbf K\\ (\\text{{eV/Å}}^2)}}"
        f"\\underbrace{{{_bmatrix(_u_full)}}}_{{\\mathbf u\\ (\\text{{Å}})}}"
        f"=\\underbrace{{{_bmatrix(_f_full)}}}_{{\\mathbf f\\ (\\text{{eV/Å}})}}$$\n\n"
        + _reduced_text
    )
    return


@app.cell
def solve_chain(F_applied, K, K_ff, b, free, np, solvable, u_set):
    if solvable:
        u = u_set.copy()
        if free.any():
            u[free] = np.linalg.solve(K_ff, b)
        # With all displacements known, each clamped row of K u gives its clamp force.
        external = K @ u
        residual = K_ff @ u[free] - b
    else:
        u = external = residual = None
    return external, residual, u


@app.cell(hide_code=True)
def solution_report(F_applied, clamped, external, k, mo, n, n_free, np, rank, residual, solvable, u):
    if not solvable:
        _total = F_applied.sum()
        _report = mo.md(
            "### Solution\n\n"
            f"The {n_free} × {n_free} matrix for the free atoms has rank **{rank}**, "
            "so `np.linalg.solve` cannot invert it. Shifting every atom by the same "
            "amount stretches no spring and leaves every balance equation unchanged. "
            "No clamp holds the chain in place, so the equations cannot decide "
            "where it sits.\n\n"
            f"The applied forces add up to **{_total:g} eV/Å**. "
            + (
                "Because they cancel, many positions are in equilibrium and any one "
                "of them satisfies the equations."
                if abs(_total) < 1e-12
                else "Because they do not cancel, no static arrangement exists: the "
                "whole chain accelerates, and its mass would matter after all."
            )
            + " Clamp at least one atom to obtain a unique answer."
        )
    else:
        _tension = k * np.diff(u)
        _atom_rows = [
            "| Atom | Condition | $u_i$ (Å) | External force (eV/Å) |",
            "|---:|---|---:|---:|",
        ]
        for _i in range(n):
            _kind = "clamped" if clamped[_i] else "free"
            _label = "$R" if clamped[_i] else "$F"
            _atom_rows.append(
                f"| {_i + 1} | {_kind} | {u[_i]:.4f} | {_label}_{{{_i + 1}}}$ = {external[_i]:.4f} |"
            )
        _spring_rows = [
            "| Spring | $k_i$ (eV/Å²) | Stretch $u_{i+1}-u_i$ (Å) | Tension $T_i$ (eV/Å) |",
            "|---:|---:|---:|---:|",
        ]
        for _i in range(n - 1):
            _spring_rows.append(
                f"| {_i + 1} | {k[_i]:g} | {u[_i + 1] - u[_i]:.4f} | {_tension[_i]:.4f} |"
            )
        _res = float(np.max(np.abs(residual))) if residual.size else 0.0
        _report = mo.md(
            "### Solution and checks\n\n"
            "`np.linalg.solve` returns the free displacements. With every $u_i$ "
            "known, the clamped rows of $\\mathbf K\\mathbf u$ give the clamp forces.\n\n"
            + "\n".join(_atom_rows)
            + "\n\n"
            + "\n".join(_spring_rows)
            + "\n\n"
            f"- Largest residual of the reduced system: **{_res:.1e} eV/Å**\n"
            f"- Sum of all external forces on the chain: **{external.sum():.1e} eV/Å**. "
            "The springs only exchange forces between atoms, so applied and clamp "
            "forces together must balance."
        )
    _report
    return


@app.cell(hide_code=True)
def plot_controls(mo):
    magnify = mo.ui.slider(1, 10, value=4, step=1, label="Displacement magnification in the sketch", show_value=True)
    magnify
    return (magnify,)


@app.cell(hide_code=True)
def chain_figure(F_applied, TwoSlopeNorm, clamped, external, k, magnify, n, np, plt, solvable, u):
    spacing = 2.5  # Å, rest distance between neighbours in the sketch

    def _zigzag(x0, x1, y, turns=6, height=0.12):
        _x = np.linspace(x0, x1, 2 * turns + 3)
        _y = np.full_like(_x, y)
        _y[1:-1:2] += height
        _y[2:-1:2] -= height
        _y[1] = _y[-2] = y
        return _x, _y

    chain_fig, (_ax, _ax_u) = plt.subplots(
        2, 1, figsize=(9, 5.2), layout="constrained", height_ratios=[1.3, 1]
    )
    _x_rest = spacing * np.arange(n)
    _ax.scatter(_x_rest, np.full(n, 0.7), s=110, facecolor="none", edgecolor="0.6", zorder=3)
    for _i in range(n - 1):
        _ax.plot(*_zigzag(_x_rest[_i] + 0.2, _x_rest[_i + 1] - 0.2, 0.7), color="0.75", lw=1)
    _ax.text(_x_rest[0] - 1.3, 0.7, "rest", va="center", ha="right", color="0.45")

    if solvable:
        _x_new = _x_rest + magnify.value * u
        _tension = k * np.diff(u)
        _limit = max(float(np.max(np.abs(_tension))), 1e-6)
        _norm = TwoSlopeNorm(vcenter=0.0, vmin=-_limit, vmax=_limit)
        _cmap = plt.get_cmap("coolwarm")
        for _i in range(n - 1):
            _ax.plot(
                *_zigzag(_x_new[_i] + 0.2, _x_new[_i + 1] - 0.2, 0.0),
                color=_cmap(_norm(_tension[_i])), lw=2.2,
            )
        _ax.scatter(_x_new[~clamped], np.zeros((~clamped).sum()), s=140, color="tab:blue", zorder=4, label="free atom")
        _ax.scatter(_x_new[clamped], np.zeros(clamped.sum()), s=140, marker="s", color="0.2", zorder=4, label="clamped atom")
        _arrow_scale = 1.6 / max(float(np.max(np.abs(external))), 1e-6)
        for _i in range(n):
            if abs(external[_i]) < 1e-9:
                continue
            # Applied forces sit on one row and clamp forces on the row below.
            _y = -0.7 if clamped[_i] else -0.35
            _ax.annotate(
                "", xy=(_x_new[_i] + _arrow_scale * external[_i], _y), xytext=(_x_new[_i], _y),
                arrowprops=dict(arrowstyle="-|>", lw=1.8,
                                color="0.2" if clamped[_i] else "tab:green",
                                ls="--" if clamped[_i] else "-"),
            )
            _ax.text(_x_new[_i] - 0.15 * np.sign(external[_i]), _y,
                     f"{'R' if clamped[_i] else 'F'}$_{{{_i + 1}}}$",
                     ha="right" if external[_i] > 0 else "left", va="center", fontsize=9)
        _ax.text(_x_rest[0] - 1.3, 0.0, "loaded", va="center", ha="right", color="0.2")
        _sm = plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)
        chain_fig.colorbar(_sm, ax=_ax, label="Spring tension (eV/Å)", shrink=0.8)
        _ax.legend(loc="upper right", fontsize=8, ncols=2, frameon=False)

        _ax_u.axhline(0, color="0.5", lw=0.8)
        _ax_u.plot(np.arange(1, n + 1), u, "o-", color="tab:blue")
        _ax_u.scatter(np.flatnonzero(clamped) + 1, u[clamped], marker="s", s=60, color="0.2", zorder=4)
        _ax_u.set(xlabel="Atom number $i$", ylabel="Displacement $u_i$ (Å)", xticks=np.arange(1, n + 1))
        _ax_u.grid(alpha=0.25)
    else:
        _ax.text(np.mean(_x_rest), 0.0, "No unique equilibrium: clamp at least one atom",
                 ha="center", va="center", color="tab:red")
        _ax_u.set_axis_off()

    _ax.set(xlim=(_x_rest[0] - 3.5, _x_rest[-1] + 3.0), ylim=(-1.0, 1.05), yticks=[],
            xlabel="Position along the chain (Å), displacements magnified")
    _ax.set_title(
        f"Displacements drawn ×{magnify.value}; red springs are stretched, blue springs are compressed",
        fontsize=10,
    )
    chain_fig
    return (chain_fig,)


@app.cell(hide_code=True)
def things_to_try(mo):
    mo.md(r"""
    ### Questions to explore

    1. In *Pull the right end, left end clamped*, every spring carries the same
       tension. Use the force balance of the last atom, then of its neighbour,
       to explain why. How does the clamp force $R_1$ compare with $F_N$?
    2. Make one spring ten times stiffer than the others. Which stretches change,
       and which stay the same? Relate the result to springs in series.
    3. In *Stretch between two clamped ends*, no force is applied, yet the atoms
       move. Where does the right-hand side of the reduced system come from?
    4. Choose *Pull the right end, no atom clamped* and read the rank of the
       matrix. Then apply equal and opposite forces to the two end atoms. Why
       does the solver still fail even though the chain is now balanced?
    5. Add up the entries in any row of $\mathbf K$. What does the result say
       about moving the whole chain rigidly?
    """)
    return


if __name__ == "__main__":
    app.run()
