# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "ase>=3.26", "weas-widget>=0.2.6"]
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
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.calculators.lj import LennardJones
    from ase.cluster import Icosahedron
    from ase.optimize import BFGS, FIRE, LBFGS
    from weas_widget import WeasWidget

    return Atoms, BFGS, EMT, FIRE, Icosahedron, LBFGS, LennardJones, WeasWidget, mo, np, plt, time


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L14 · Relaxing a cluster with an atomistic package

    The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) organizes
    a relaxation into three objects:

    $$
    \boxed{\text{structure (Atoms)}}\;\rightarrow\;
    \boxed{\text{calculator: }E,\ \mathbf F}\;\rightarrow\;
    \boxed{\text{optimizer}}\;\rightarrow\;
    \boxed{\text{relaxed structure}}
    $$

    The `Atoms` object stores the elements and coordinates. The calculator returns
    the energy and the force on every atom. The optimizer moves the atoms until the
    largest force component is below `fmax`. The same optimizer works with any
    calculator, whether it is a Lennard-Jones pair potential, the effective-medium
    theory (EMT) model for metals used in the L11 copper animation, or a quantum
    mechanical method.
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    material = mo.ui.dropdown({"Argon, Lennard-Jones calculator": "Ar",
                               "Copper, EMT calculator": "Cu"},
                              value="Argon, Lennard-Jones calculator", label="Material and calculator")
    n_atoms = mo.ui.slider(4, 30, value=13, step=1, label="Number of atoms", show_value=True,
                           debounce=True)
    shake = mo.ui.slider(0.05, 0.5, value=0.3, step=0.05, label="Random displacement (Å)",
                         show_value=True, debounce=True)
    fmax = mo.ui.dropdown({"0.05 eV/Å": 0.05, "0.01 eV/Å": 0.01, "0.001 eV/Å": 0.001},
                          value="0.001 eV/Å", label="Force tolerance fmax")
    mo.hstack([material, n_atoms, shake, fmax], justify="start", gap=1, wrap=True)
    return fmax, material, n_atoms, shake


@app.cell(hide_code=True)
def step_one_text(mo):
    mo.md(r"""
    ### Live step 1 · A structure and its calculator

    We take the atoms nearest the centre of a 55-atom icosahedron, shift each atom
    randomly by up to the chosen displacement, and attach a calculator. For argon,
    `LennardJones(epsilon=0.0103, sigma=3.40, rc=...)` uses the A1 parameters. A
    large cutoff `rc` keeps every pair, as in our own cluster energy.

    ```python
    atoms = Atoms("Ar13", positions=X)          # X: (13, 3) array in Å
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40, rc=50.0)
    atoms.get_potential_energy(), atoms.get_forces()
    ```
    """)
    return


@app.cell
def live_structure(Atoms, EMT, Icosahedron, LennardJones, np):
    # Live step 1: build a displaced cluster and attach the matching calculator.
    def make_cluster(symbol, N, shake, seed=374):
        nn = 3.40*2**(1/6) if symbol == "Ar" else 2.55          # nearest-neighbour distance, Å
        seed_cluster = Icosahedron(symbol, noshells=3, latticeconstant=np.sqrt(2)*nn)
        centre = seed_cluster.positions.mean(axis=0)
        nearest = np.argsort(np.linalg.norm(seed_cluster.positions - centre, axis=1))[:N]
        X = seed_cluster.positions[nearest] - centre
        X = X + np.random.default_rng(seed).uniform(-shake, shake, X.shape)
        atoms = Atoms(symbol*N, positions=X)
        if symbol == "Ar":
            atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40, rc=50.0)
        else:
            atoms.calc = EMT()
        return atoms

    return (make_cluster,)


@app.cell(hide_code=True)
def step_two_text(mo):
    mo.md(r"""
    ### Live step 2 · Run three optimizers from the same structure

    `BFGS` builds an approximate Hessian, as in `scipy.optimize.minimize`. `LBFGS`
    stores only recent steps. `FIRE` (fast inertial relaxation engine) moves the atoms
    like a damped molecular dynamics simulation. Attaching a function to the optimizer
    records the energy and the largest force after each step.

    ```python
    opt = BFGS(atoms, logfile=None)
    opt.attach(record)          # called after every step
    opt.run(fmax=0.001)         # stop when every |force component| < fmax (eV/Å)
    ```

    Docs: [`ase.optimize`](https://wiki.fysik.dtu.dk/ase/ase/optimize.html)
    """)
    return


@app.cell
def live_relax(BFGS, FIRE, LBFGS, np):
    # Live step 2: relax a copy of the structure; return the relaxed atoms and the step history.
    OPTIMIZERS = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}

    def run_optimizer(atoms, name, fmax, max_steps=2000):
        """Relax atoms in place with the named optimizer; return them and (E, max|F|) per step."""
        history = []
        record = lambda: history.append((atoms.get_potential_energy(),
                                         np.abs(atoms.get_forces()).max()))
        opt = OPTIMIZERS[name](atoms, logfile=None)
        opt.attach(record)
        opt.run(fmax=fmax, steps=max_steps)
        return atoms, np.array(history)

    return OPTIMIZERS, run_optimizer


@app.cell(hide_code=True)
def run_all(OPTIMIZERS, fmax, make_cluster, material, n_atoms, run_optimizer, shake, time):
    start_atoms = make_cluster(material.value, n_atoms.value, shake.value)
    E_start = start_atoms.get_potential_energy()
    relaxed, histories, seconds = {}, {}, {}
    for _name in OPTIMIZERS:
        _t0 = time.perf_counter()
        # A fresh copy of the same displaced start (same random seed) for each optimizer.
        _atoms = make_cluster(material.value, n_atoms.value, shake.value)
        relaxed[_name], histories[_name] = run_optimizer(_atoms, _name, fmax.value)
        seconds[_name] = time.perf_counter() - _t0
    return E_start, histories, relaxed, seconds, start_atoms


@app.cell(hide_code=True)
def plot_history(E_start, fmax, histories, material, mo, np, plt, seconds):
    def draw():
        E_final = min(h[-1, 0] for h in histories.values())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
        styles = {"BFGS": dict(ls="--", lw=2.5, zorder=3), "LBFGS": dict(lw=1.5), "FIRE": dict(lw=1.5)}
        for name, h in histories.items():
            steps = np.arange(len(h))
            ax1.semilogy(steps, np.maximum(h[:, 0] - E_final, 1e-12), label=name, **styles[name])
            ax2.semilogy(steps, h[:, 1], label=name, **styles[name])
        ax2.axhline(fmax.value, color="black", ls="--", lw=0.8, label="fmax")
        ax1.set(xlabel="Optimizer step", ylabel="E − E_final (eV)", title="Energy above the final value")
        ax2.set(xlabel="Optimizer step", ylabel="Largest |force component| (eV/Å)",
                title="Largest force")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    history_figure = draw()
    _rows = [f"| {n} | {len(h) - 1} | {h[-1, 0]:.6f} | {h[-1, 1]:.1e} | {1000*seconds[n]:.0f} |"
             for n, h in histories.items()]
    mo.vstack([history_figure, mo.md("\n".join([
        f"Starting energy **{E_start:.4f} eV** ({material.value}).",
        "",
        "| Optimizer | Steps | Final energy (eV) | Largest force (eV/Å) | Time (ms) |",
        "|---|---:|---:|---:|---:|", *_rows,
        "",
        "For copper, EMT energies are measured relative to bulk fcc copper, so a "
        "cluster has a positive energy. Only energy differences are meaningful."]))])
    return (history_figure,)


@app.cell(hide_code=True)
def show_structures(WeasWidget, mo, relaxed, start_atoms):
    def viewer(atoms, title):
        positions = atoms.positions - atoms.positions.mean(axis=0)
        view = atoms.copy()
        view.positions = positions
        w = WeasWidget(from_ase=view, viewerStyle={"width": "100%", "height": "320px"},
                       modelStyle=1, cellSettings={"showCell": False},
                       cameraSetting={"lookAt": [0, 0, 0], "direction": [1, 0.6, 0.4],
                                      "distance": 40, "zoom": 1})
        return mo.vstack([mo.md(f"**{title}**"), mo.ui.anywidget(w.children[0])])

    mo.vstack([
        mo.hstack([viewer(start_atoms, "Displaced start"), viewer(relaxed["BFGS"], "After BFGS")],
                  widths="equal", align="start"),
        mo.md("Drag to rotate and scroll to zoom. Viewer edits do not change the calculation."),
    ])
    return


if __name__ == "__main__":
    app.run()
