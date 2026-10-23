# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.13"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.optimize import minimize

    return minimize, mo, np


@app.cell
def _(np):
    # Lennard-Jones cluster in reduced units; q holds (x1, y1, z1, x2, ..., zN)
    def lj_energy(q):
        X = q.reshape(-1, 3)
        i, j = np.triu_indices(len(X), k=1)
        r = np.linalg.norm(X[i] - X[j], axis=1)
        return np.sum(4 * (r**-12 - r**-6))


    def lj_gradient(q):
        X = q.reshape(-1, 3)
        i, j = np.triu_indices(len(X), k=1)
        d = X[i] - X[j]
        r = np.linalg.norm(d, axis=1)
        dV_dr = 24 * (r**-7 - 2 * r**-13)
        pair = (dV_dr / r)[:, None] * d
        g = np.zeros_like(X)
        np.add.at(g, i, pair)
        np.add.at(g, j, -pair)
        return g.ravel()

    return lj_energy, lj_gradient


@app.cell
def _(np):
    def forward_gradient(E, q, h=1e-8):
        g = np.empty_like(q)
        E0 = E(q)                          # shared by every component
        for j in range(q.size):
            dq = np.zeros_like(q)
            dq[j] = h
            g[j] = (E(q + dq) - E0) / h
        return g


    def central_gradient(E, q, h=1e-5):
        g = np.empty_like(q)
        for j in range(q.size):
            dq = np.zeros_like(q)
            dq[j] = h
            g[j] = (E(q + dq) - E(q - dq)) / (2 * h)
        return g

    return central_gradient, forward_gradient


@app.cell
def _(mo):
    n_choice = mo.ui.dropdown(options=["4", "8", "27"], value="4", label="atoms N")
    return (n_choice,)


@app.cell
def _(n_choice, np):
    # Start from a square (N = 4) or a cube of atoms 1.2 σ apart, slightly distorted
    N = int(n_choice.value)
    side = {4: (2, 2, 1), 8: (2, 2, 2), 27: (3, 3, 3)}[N]
    grid = np.array([(a, b, c) for a in range(side[0]) for b in range(side[1]) for c in range(side[2])], float)
    q0 = (1.2 * grid + np.random.default_rng(1).normal(0.0, 0.05, grid.shape)).ravel()
    return N, q0


@app.cell
def _(N, central_gradient, forward_gradient, lj_energy, lj_gradient, minimize, mo, n_choice, np, q0):
    def relax(method):
        calls = {"energy": 0, "in gradients": 0}

        def E(q):
            calls["energy"] += 1
            return lj_energy(q)

        def jac(q):
            before = calls["energy"]
            if method == "analytical":
                g = lj_gradient(q)
            elif method == "forward":
                g = forward_gradient(E, q)
            else:
                g = central_gradient(E, q)
            calls["in gradients"] += calls["energy"] - before
            return g

        res = minimize(E, q0, jac=jac, method="BFGS", options={"gtol": 1e-4})
        true_force = np.max(np.abs(lj_gradient(res.x)))
        return res, calls, true_force


    d = 3 * N
    rows = [
        "| gradient | BFGS steps | gradients | energy calls | in gradients | final $E$ | max force |",
        "|---|---|---|---|---|---|---|",
    ]
    for _method in ["analytical", "forward", "central"]:
        _res, _calls, _F = relax(_method)
        rows.append(
            f"| {_method} | {_res.nit} | {_res.njev} | {_calls['energy']} | {_calls['in gradients']}"
            f" | {_res.fun:.4f} | {_F:.1e} |"
        )
    mo.vstack(
        [
            mo.hstack(
                [n_choice, mo.md(f"$d=3N={d}$ coordinates: forward gradient {d + 1} energies, central gradient {2 * d} energies")],
                justify="start",
                align="center",
                gap=2,
                widths=[1, 4],
            ),
            mo.md("\n".join(rows)),
        ]
    )
    return



if __name__ == "__main__":
    app.run()
