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
    from scipy.optimize import root
    return mo, np, plt, root


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## How far can we pull three atoms?

    The middle atom is fixed at zero. The left and right atoms, at $x_1<0<x_2$,
    feel applied forces $-f$ and $+f$. All three pairs interact through the LJ energy.
    Length, energy, and force are measured in $\sigma$, $\varepsilon$, and
    $\varepsilon/\sigma$.

    **Predict:** will doubling the pull double the extension? Start at $f=0$, then
    increase the load toward 2.44. Inspect the positions and total bond energy
    at individual Newton steps.
    """)
    return


@app.cell
def pair_model():
    def V(r):
        return 4*(r**-12 - r**-6)

    def dV(r):
        return 24*(r**-7 - 2*r**-13)

    def d2V(r):
        return 24*(26*r**-14 - 7*r**-8)

    r0 = 2**(1/6)
    r_s = (26/7)**(1/6)
    return V, d2V, dV, r0, r_s


@app.cell(hide_code=True)
def force_text(mo):
    mo.md(r"""
    ### Energy, forces, and the 2 × 2 Jacobian

    $$E=V(-x_1)+V(x_2)+V(x_2-x_1),$$
    $$F_1=-f+V'(-x_1)+V'(x_2-x_1),\qquad
      F_2=f-V'(x_2)-V'(x_2-x_1).$$

    Each diagonal derivative is negative local stiffness. The outer-pair interaction
    gives the off-diagonal entries $+V''(x_2-x_1)$, coupling the two moving atoms.
    """)
    return


@app.cell
def trimer_model(V, d2V, dV, np):
    def energy(x):
        x1, x2 = x
        return V(-x1) + V(x2) + V(x2-x1)

    def force(x, f):
        x1, x2 = x
        return np.array([-f + dV(-x1) + dV(x2-x1),
                          f - dV(x2) - dV(x2-x1)])

    def force_jacobian(x, f=0.0):
        x1, x2 = x
        k_left, k_right, k_outer = d2V(-x1), d2V(x2), d2V(x2-x1)
        return np.array([[-k_left-k_outer, k_outer],
                         [k_outer, -k_right-k_outer]])

    def ordered_trimer(x):
        return x[0] < -0.5 and x[1] > 0.5

    return energy, force, force_jacobian, ordered_trimer


@app.cell(hide_code=True)
def method_text(mo):
    mo.md(r"""
    ### Newton's method

    Solve $\mathbf J\Delta\mathbf x=-\mathbf F$ and add the correction to the positions.
    Each history entry records the position, net force, Jacobian, and proposed correction.
    We accept a point when both the residual and correction are small. A guard stops a
    step if a neighbouring separation would fall below $0.5\sigma$, where LJ repulsion
    is enormous. It reports failure rather than silently moving the atoms elsewhere.
    """)
    return


@app.cell
def newton_algorithm(np):
    def newton_system(F, J, x0, valid=lambda x: True,
                      ftol=1e-9, xtol=1e-10, max_steps=40):
        x = np.array(x0, dtype=float)
        history = []
        for m in range(max_steps + 1):
            if not np.all(np.isfinite(x)) or not valid(x):
                return x, history, False, "Inadmissible positions. Try a closer initial guess."
            f = F(x)
            matrix = J(x)
            if not (np.all(np.isfinite(f)) and np.all(np.isfinite(matrix))):
                return x, history, False, "Non-finite force or Jacobian."
            try:
                delta = np.linalg.solve(matrix, -f)
            except np.linalg.LinAlgError:
                history.append((x.copy(), f, matrix, None))
                return x, history, False, "Singular Jacobian."
            history.append((x.copy(), f, matrix, delta))
            if np.max(np.abs(f)) < ftol and np.max(np.abs(delta)) < xtol:
                return x, history, True, "Converged"
            if m == max_steps:
                break
            x = x + delta
        return x, history, False, "Maximum number of Newton steps reached."

    return (newton_system,)


@app.cell(hide_code=True)
def controls(mo):
    f_pull = mo.ui.slider(0.0, 2.6, value=2.0, step=0.001,
                          label="Outward pull f (ε/σ)", show_value=True,
                          include_input=True, full_width=True, debounce=True)
    guess_left = mo.ui.number(-2.5, -0.6, value=-1.12, step=0.01, label="Initial x₁ (σ)")
    guess_right = mo.ui.number(0.6, 2.5, value=1.12, step=0.01, label="Initial x₂ (σ)")
    continuation = mo.ui.checkbox(value=True, label="Raise load in steps from zero (continuation)")
    return continuation, f_pull, guess_left, guess_right


@app.cell
def loading(force, force_jacobian, newton_system, np, ordered_trimer):
    def solve_trimer(x0, f):
        return newton_system(lambda x: force(x, f), lambda x: force_jacobian(x, f),
                             x0, valid=ordered_trimer)

    def load_trimer(x0, target, use_continuation=True):
        x = np.array(x0, dtype=float)
        loads = np.linspace(0, target, max(2, int(np.ceil(target/0.05))+1)) if use_continuation else [target]
        records = []
        for load in loads:
            start = x.copy()
            x, history, ok, message = solve_trimer(start, load)
            records.append((float(load), x.copy(), ok))
            if not ok:
                return x, history, ok, message, records, start
        return x, history, ok, message, records, start

    return load_trimer, solve_trimer


@app.cell(hide_code=True)
def calculation(continuation, f_pull, guess_left, guess_right, load_trimer):
    chain_x, history, converged, status, load_records, final_start = load_trimer(
        [guess_left.value, guess_right.value], f_pull.value, continuation.value)
    solved_load = load_records[-1][0]
    return chain_x, converged, final_start, history, load_records, solved_load, status


@app.cell(hide_code=True)
def iteration_control(history, mo):
    iteration = mo.ui.number(0, max(0, len(history)-1), value=max(0, len(history)-1), step=1,
                             label="Iteration m")
    return (iteration,)


@app.cell(hide_code=True)
def controls_display(continuation, f_pull, guess_left, guess_right, iteration, mo):
    mo.vstack([
        f_pull,
        mo.hstack([guess_left, guess_right, iteration], justify="start", gap=2),
        continuation,
    ])
    return


@app.cell(hide_code=True)
def visualization(converged, energy, history, iteration, mo, np, plt, solved_load, status):
    mo.stop(not history, mo.callout(status, kind="warn"))
    _m = min(iteration.value, len(history)-1)
    _x, _f, _j, _delta = history[_m]
    _path = np.array([row[0] for row in history])
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 12})
    chain_figure, _axes = plt.subplots(1, 2, figsize=(7.1, 3.8), layout="constrained")
    _pos = np.array([_x[0], 0, _x[1]])
    _axes[0].plot(_pos, [0, 0, 0], color="#8a8f98", lw=3, zorder=1)
    _axes[0].scatter(_pos, [0, 0, 0], s=400, c=["#b5473a", "#8a8f98", "#b5473a"], edgecolors="#231f20", zorder=2)
    for _p, _label in zip(_pos, [f"$x_1$ = {_x[0]:.4f}", "fixed: 0", f"$x_2$ = {_x[1]:.4f}"]):
        _axes[0].text(_p, -0.16 if _p != 0 else -0.28, _label, ha="center", fontsize=10)
    for _i, _direction in [(0, -1), (2, 1)]:
        _axes[0].annotate("", xy=(_pos[_i]+_direction*0.55, 0),
                          xytext=(_pos[_i]+_direction*0.12, 0),
                          arrowprops={"arrowstyle": "->", "color": "#b5473a", "lw": 2})
    _axes[0].text(0, 0.22, f"outward load f = {solved_load:.3f}", ha="center")
    _axes[0].set(xlim=(min(-2, _x[0]-0.7), max(2, _x[1]+0.7)), ylim=(-0.4, 0.4),
                 title=f"Positions: iteration {_m}", xlabel="Position (σ)")
    _axes[0].set_yticks([])
    _axes[0].spines[["top", "right", "left"]].set_visible(False)
    _a, _b = np.meshgrid(np.linspace(-1.65, -1.0, 160), np.linspace(1.0, 1.65, 160))
    _e = energy((_a, _b))
    _cs = _axes[1].contourf(_a, _b, _e, levels=20, cmap="Greys_r")
    chain_figure.colorbar(_cs, ax=_axes[1], label="Total bond energy E (ε)")
    _axes[1].plot(_path[:, 0], _path[:, 1], ".--", color="#b5473a", ms=6)
    _axes[1].plot(_x[0], _x[1], "o", color="#b5473a", ms=8)
    _axes[1].set(xlim=(-1.65, -1.0), ylim=(1.0, 1.65), xlabel="$x_1$ (σ)", ylabel="$x_2$ (σ)",
                 title="Internal energy")
    for _label, _ax in zip("ab", _axes):
        _ax.text(-0.12, 1.06, _label, transform=_ax.transAxes, weight="bold", fontsize=18)
        _ax.set_title(_ax.get_title(), fontsize=12)
        _ax.tick_params(labelsize=10)
    mo.vstack([
        chain_figure,
        mo.callout(status,
                   kind="success" if converged else "warn"),
    ])
    return (chain_figure,)


@app.cell(hide_code=True)
def history_table(energy, history, mo, np):
    mo.ui.table([
        {"m": m, "x₁ (σ)": f"{row[0][0]:.6f}", "x₂ (σ)": f"{row[0][1]:.6f}",
         "E (ε)": f"{energy(row[0]):.6f}", "max |Fᵢ| (ε/σ)": f"{np.max(np.abs(row[1])):.3e}",
         "max |Δxᵢ| (σ)": "—" if row[3] is None else f"{np.max(np.abs(row[3])):.3e}"}
        for m, row in enumerate(history)
    ], selection=None)
    return


@app.cell
def library_solve(final_start, force, force_jacobian, np, root, solved_load):
    library_result = root(force, final_start, args=(solved_load,), jac=force_jacobian, method="hybr")
    library_residual = np.max(np.abs(force(library_result.x, solved_load)))
    return library_residual, library_result


@app.cell(hide_code=True)
def library_check(library_residual, library_result, mo, np, ordered_trimer, solved_load):
    library_admissible = bool(
        np.all(np.isfinite(library_result.x)) and ordered_trimer(library_result.x))
    library_accepted = bool(
        library_result.success and library_admissible and library_residual < 1e-9)
    mo.vstack([
        mo.md(f"""
        ### SciPy at the same load and starting point

        Load: **{solved_load:.5f} ε/σ**. Coordinates: **{library_result.x} σ**.

        - Solver success flag: **{bool(library_result.success)}**
        - Largest net force: **{library_residual:.3e} ε/σ** (target: below 1e-9)
        - Ordered coordinates with separations above 0.5 σ: **{library_admissible}**
        """),
        mo.callout(str(library_result.message),
                   kind="success" if library_accepted else "warn"),
        mo.md("""
        A small residual establishes force balance at these coordinates.
        At a failed load, reduce the load increment and compare with the
        force–separation maximum before interpreting the failure as a load limit.
        """),
    ])
    return library_accepted, library_admissible


@app.cell
def general_chain(V, d2V, dV, np):
    def chain_model(n):
        if n < 3 or n % 2 == 0:
            raise ValueError("Choose an odd number of atoms, at least three.")
        fixed = n//2
        free = np.delete(np.arange(n), fixed)

        def positions(x):
            pos = np.zeros(n)
            pos[free] = x
            return pos

        def evaluate(x, f):
            pos = positions(x)
            forces = np.zeros(n)
            J = np.zeros((n, n))
            E = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    r = pos[j] - pos[i]
                    E += V(r)
                    pair_force, k = dV(r), d2V(r)
                    forces[i] += pair_force
                    forces[j] -= pair_force
                    J[i, i] -= k
                    J[j, j] -= k
                    J[i, j] += k
                    J[j, i] += k
            forces[0] -= f
            forces[-1] += f
            return E, forces[free], J[np.ix_(free, free)]

        def valid(x):
            return bool(np.all(np.diff(positions(x)) > 0.5))

        return positions, evaluate, valid

    return (chain_model,)


@app.cell(hide_code=True)
def seven_atom_text(mo):
    mo.md(r"""
    ### Extension: seven atoms

    Hold atom 3 at zero and pull the two ends equally. There are six moving
    coordinates. The function above adds the energy, equal-and-opposite forces,
    and Jacobian entries of every pair before selecting the free coordinates.

    **Predict:** will the end bonds and interior bonds stretch equally?
    Edit `seven_load` in the next cell and compare the bond strains.
    """)
    return


@app.cell
def seven_atom_calculation(chain_model, newton_system, np, r0):
    seven_load = 2.0  # outward force in ε/σ; try 0, 2.43, and 2.45
    positions7, evaluate7, valid7 = chain_model(7)
    x7 = r0 * np.array([-3., -2., -1., 1., 2., 3.])
    rest7, _, rest_ok7, rest_message7 = newton_system(
        lambda x: evaluate7(x, 0)[1], lambda x: evaluate7(x, 0)[2],
        x7, valid=valid7)
    x7 = rest7.copy()
    ok7, message7 = rest_ok7, rest_message7
    reached7 = 0.0
    if rest_ok7:
        for load7 in np.linspace(0, seven_load, max(2, int(np.ceil(seven_load/0.02))+1)):
            trial7, history7, ok7, message7 = newton_system(
                lambda x: evaluate7(x, load7)[1], lambda x: evaluate7(x, load7)[2],
                x7, valid=valid7)
            if not ok7:
                break
            x7 = trial7
            reached7 = float(load7)
    seven_residual = float(np.max(np.abs(evaluate7(x7, reached7)[1])))
    seven_strains = np.diff(positions7(x7))/np.diff(positions7(rest7))-1
    return (
        message7, ok7, reached7, rest_ok7, seven_load, seven_residual, seven_strains,
    )


@app.cell(hide_code=True)
def seven_atom_result(message7, mo, ok7, reached7, rest_ok7, seven_load, seven_residual, seven_strains):
    mo.stop(not rest_ok7, mo.callout(message7, kind="warn"))
    mo.vstack([
        mo.md(f"""
        Requested load: **{seven_load:.5f} ε/σ**.
        Last accepted load: **{reached7:.5f} ε/σ**.
        Largest force at that accepted state: **{seven_residual:.3e} ε/σ**.
        """),
        mo.callout(message7 if ok7 else
                   "The next load step failed. Results below are for the last accepted load. " + message7,
                   kind="success" if ok7 else "warn"),
        mo.ui.table([
            {"Bond (left to right)": i+1, "Strain (%)": f"{100*strain:.3f}"}
            for i, strain in enumerate(seven_strains)
        ], selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
