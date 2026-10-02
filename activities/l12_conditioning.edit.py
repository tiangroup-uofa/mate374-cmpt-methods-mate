# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    ## How does a weak anchor change the displacement?
    Atom 0 is fixed; springs connect 0–1 and 1–2. Stiffnesses, forces,
    and displacements are numbers in one consistent set of units.
    **Predict:** compare `k01 = 5.0` with `k01 = 0.001`.
    Then try `k01 = 0.0`. Which motion loses its restoring force?
    """)
    return


@app.cell
def _(np):
    k01 = 0.001
    k12 = 5.0
    f = np.array([0.0, 1.0])
    K = np.array([[k01 + k12, -k12], [-k12, k12]])
    print("K =\n", K)
    print("det(K) =", np.linalg.det(K))
    print("2-norm condition number =", np.linalg.cond(K))
    try:
        u = np.linalg.solve(K, f)
        print("solve: displacements =", u)
        print("inverse @ f =", np.linalg.inv(K) @ f)
        print("Residual =", K @ u - f)
    except np.linalg.LinAlgError:
        print("Singular K: no unique displacement. Try f = [-1.0, 1.0].")
    return (K,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A small relative force perturbation
    Equal and opposite forces initially stretch only spring 1–2.
    Add a small net force at atom 2. **Predict:** will the pair translate
    noticeably when its anchor is weak? Compare relative changes below.
    A small residual checks the equations; sensitivity measures how much
    their solution changes when the supplied forces change.
    """)
    return


@app.cell
def _(K, np):
    f_balanced = np.array([-1.0, 1.0])
    delta_f = np.array([0.0, 0.001])
    try:
        u_balanced = np.linalg.solve(K, f_balanced)
        u_changed = np.linalg.solve(K, f_balanced + delta_f)
        relative_force = np.linalg.norm(delta_f) / np.linalg.norm(f_balanced)
        relative_u = np.linalg.norm(u_changed - u_balanced) / np.linalg.norm(u_balanced)
        print("Original displacements:", u_balanced)
        print("Perturbed displacements:", u_changed)
        print(f"Relative force change: {relative_force:.6g}")
        print(f"Relative displacement change: {relative_u:.6g}")
        print(f"Condition-number bound: {np.linalg.cond(K) * relative_force:.6g}")
    except np.linalg.LinAlgError:
        print("The sensitivity bound requires a nonsingular matrix.")
    return


@app.cell
def _(np):
    # Edit these matrices to test the in-class questions.
    A = np.array([[1.0, 1.0], [1.0, 1.001]])
    B = np.array([[0.003, 3.0], [1.0, 1.0]])
    for _name, _matrix in [("A", A), ("B", B), ("small identity", 1e-8 * np.eye(2))]:
        print(_name, "det =", np.linalg.det(_matrix), "cond =", np.linalg.cond(_matrix))
    return


if __name__ == "__main__":
    app.run()
