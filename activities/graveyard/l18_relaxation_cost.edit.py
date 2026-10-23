# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## How many energy evaluations does a relaxation need?
    Model $N$ atoms tethered to fixed sites by independent harmonic springs:
    $E=E_0+(k_1q_1^2+\cdots+k_dq_d^2)/2$, with $d=3N$.
    The exact gradient is $(k_1q_1,\ldots,k_dq_d)$.
    **Predict:** how does doubling $N$ change the energy calls for one gradient?
    After comparing forward and central differences, change `energy_offset`
    to `-1e8` and `-1e12`. Should a constant change the equilibrium?
    """)
    return


@app.cell
def _(np):
    n_atoms = 10
    d = 3 * n_atoms
    h = 1e-4                         # Å; try 1e-3 or 1e-5
    energy_offset = 0.0              # eV; try -1e8 or -1e12
    k = np.linspace(1.0, 4.0, d)     # eV / Å²
    q_initial = np.full(d, 0.1)      # Å from each fixed site
    alpha = 0.2                     # Å² / eV; alpha * max(k) < 2
    force_tolerance = 1e-6           # eV / Å, largest absolute component
    max_steps = 120
    return alpha, d, energy_offset, force_tolerance, h, k, max_steps, q_initial


@app.cell
def _(alpha, energy_offset, force_tolerance, h, k, max_steps, np):
    def relax(q_start, method):
        # The counter belongs to this run and includes every energy evaluation.
        calls = 0

        def energy(q):
            nonlocal calls
            calls += 1
            return energy_offset + 0.5 * np.dot(k * q, q)

        def gradient(q):
            if method == "Analytical":
                return k * q
            g = np.empty_like(q)
            if method == "Forward":
                base = energy(q)      # Reuse this one value for all coordinates
            for j in range(q.size):
                plus = q.copy()
                plus[j] += h
                if method == "Forward":
                    g[j] = (energy(plus) - base) / h
                else:
                    minus = q.copy()
                    minus[j] -= h
                    g[j] = (energy(plus) - energy(minus)) / (2 * h)
            return g

        q = q_start.copy()
        history = []
        gradient_evaluations = 0
        for step in range(max_steps + 1):
            g = gradient(q)
            gradient_evaluations += 1
            numerical_force = np.max(np.abs(g))
            true_force = np.max(np.abs(k * q))
            history.append(true_force)
            if numerical_force < force_tolerance or step == max_steps:
                break
            q = q - alpha * g
        return {
            "method": method,
            "steps": step,
            "gradients": gradient_evaluations,
            "calls": calls,
            "reported_force": numerical_force,
            "true_force": true_force,
            "stopped_by_tolerance": numerical_force < force_tolerance,
            "history": np.array(history),
        }

    return (relax,)


@app.cell
def _(d, q_initial, relax):
    runs = [relax(q_initial, _method) for _method in ["Analytical", "Forward", "Central"]]
    print(f"d = {d} coordinates; forward = {d + 1}, central = {2 * d} energy calls per gradient")
    print("Method       Steps  Gradients  Energy calls  Reported |F|max   True |F|max   Tolerance met?")
    for _run in runs:
        print(f"{_run['method']:12s} {_run['steps']:5d} {_run['gradients']:10d} {_run['calls']:13d}"
              f" {_run['reported_force']:17.3e} {_run['true_force']:13.3e}  {_run['stopped_by_tolerance']}")
    print("Forces in eV/Å. The analytical route still does work but needs no energy calls here.")
    return (runs,)


@app.cell
def _(force_tolerance, plt, runs):
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for _run in runs:
        ax.semilogy(_run["history"], marker="." if _run["steps"] == 0 else None, label=_run["method"])
    ax.axhline(force_tolerance, color="black", ls=":", label="Requested tolerance")
    ax.set(xlabel="Relaxation update", ylabel="True maximum force component (eV/Å)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Trust check.** The counter includes the final gradient used to test
    convergence. The fixed update avoids extra energy calls from a line search.
    For this quadratic energy, a forward difference gives
    $g_j=k_jq_j+k_jh/2$, so it drives $q_j$ towards $-h/2$.
    The central difference has zero truncation error for a quadratic.
    Compare the reported and exact forces at stopping. If adding a constant
    makes the reported force vanish immediately, the energy differences have
    fallen below the floating-point resolution.
    """)
    return


if __name__ == "__main__":
    app.run()
