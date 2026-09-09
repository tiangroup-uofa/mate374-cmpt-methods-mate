# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "ase>=3.26", "weas-widget>=0.2.6"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from ase import Atoms
    from weas_widget import WeasWidget
    return mo, np, plt, minimize, Atoms, WeasWidget


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A1 Q4 · Lennard-Jones clusters

    **Predict:** will four atoms prefer a line, a square, or a tetrahedron?
    Complete the two functions below, then use the supplied experiments.
    This is an isolated argon cluster: all unique pairs interact, with no
    cutoff or periodic boundaries. We compare potential energies, not
    finite-temperature free energies.

    $$V(r)=4\varepsilon[(\sigma/r)^{12}-(\sigma/r)^6],\qquad
    \varepsilon=0.0103\ \mathrm{eV},\quad \sigma=3.40\ \text{Å}.$$

    Open **Edit code** to complete the functions. Save your edited notebook
    before closing or resetting the browser page.
    """)
    return


@app.cell
def _():
    epsilon = 0.0103  # eV
    sigma = 3.40  # angstrom
    return epsilon, sigma


@app.cell
def _(epsilon, sigma):
    def calculate_LJ(r):
        """Return one pair energy in eV for a positive distance r in angstrom."""
        # Use epsilon and sigma above. Replace ... with your expression.
        return ...

    return (calculate_LJ,)


@app.cell(hide_code=True)
def _(mo):
    pair_points = mo.ui.slider(50, 2000, step=50, value=400,
                               label="4.2 checker: number of sampled distances", show_value=True)
    pair_points
    return (pair_points,)


@app.cell(hide_code=True)
def _(calculate_LJ, mo, np, pair_points, plt, sigma):
    mo.stop(calculate_LJ(sigma) is ..., mo.md("**4.1:** Complete `calculate_LJ` to display the pair-potential plot."))

    def plot_pair_potential():
        distances = np.linspace(3.1, 10.0, pair_points.value)
        energies = np.array([calculate_LJ(float(r)) for r in distances])
        best = int(np.argmin(energies))
        # Restrict the zero check to a sign change, not the tail approaching zero.
        crossings = np.flatnonzero(energies[:-1] * energies[1:] <= 0)
        root = None
        if len(crossings):
            bracket = int(crossings[0])
            root = bracket + int(np.argmin(np.abs(energies[bracket:bracket + 2])))
        rows = [
            {"Sampled check": "Closest to finite zero crossing", "r (Å)": None if root is None else float(distances[root]),
             "V(r) (eV)": None if root is None else float(energies[root])},
            {"Sampled check": "Currently minimal", "r (Å)": float(distances[best]),
             "V(r) (eV)": float(np.min(energies))},
        ]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(distances, energies)
        ax.plot(distances[best], energies[best], "o", label="Sampled minimum")
        if root is not None:
            ax.plot(distances[root], energies[root], "s", label="Nearest sampled zero crossing")
        ax.legend(fontsize=9)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(xlabel="Pair separation r (Å)", ylabel="Pair energy V(r) (eV)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return mo.vstack([
            fig, mo.md("### 4.2 · Sampled-point checker"), mo.ui.table(rows, selection=None),
            mo.md(f"Spacing: **{distances[1] - distances[0]:.5f} Å**. These are sampled points, "
                  "not analytical roots or minima. Increase the point count to check resolution. "
                  + ("No zero crossing was found in the sampled interval." if root is None else "")),
        ])

    plot_pair_potential()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3: add every unique pair

    Complete `calculate_LJ_cluster(coords)` using two nested loops and
    your `calculate_LJ` function. Each row of `coords` is one atom's
    `[x, y, z]` position in Å. Exclude self-pairs and count each pair once.

    $$r_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2},\qquad
    E=\sum_{i<j}V(r_{ij}).$$
    """)
    return


@app.cell
def _(calculate_LJ, np):
    def calculate_LJ_cluster(coords):
        """Return total energy in eV for an (N, 3) list or array of positions."""
        # Calculate pair distances and sum calculate_LJ(r) for i < j.
        return ...

    return (calculate_LJ_cluster,)


@app.cell(hide_code=True)
def _(calculate_LJ, calculate_LJ_cluster, mo, np, sigma):
    benchmark_coords = np.array([[0., 0., 0.], [4., 0., 0.], [0., 4.5, 0.]])
    mo.stop(
        calculate_LJ(sigma) is ... or calculate_LJ_cluster(benchmark_coords) is ...,
        mo.md("Complete both functions to run the cluster checks and experiments."),
    )
    benchmark_energy = float(calculate_LJ_cluster(benchmark_coords))
    benchmark_ok = bool(np.isclose(benchmark_energy, -0.01720978466531022, rtol=1e-6, atol=1e-10))
    mo.vstack([
        mo.md(f"**Three-atom check:** your result = {benchmark_energy:.12f} eV; expected ≈ −0.017209784665 eV."),
        mo.callout(
            "Benchmark passed. You can now explore the shapes below." if benchmark_ok
            else "Benchmark did not pass. Check the distances, units, and pair counting before interpreting the results.",
            kind="success" if benchmark_ok else "warn",
        ),
    ])
    return benchmark_coords, benchmark_ok


@app.cell(hide_code=True)
def _(Atoms, WeasWidget, mo, np, sigma):
    def four_atom_shapes(a):
        """a is nearest-neighbour spacing for the line, edge length otherwise."""
        return {
            "Line": a * np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float),
            "Square": a * np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            "Tetrahedron": a * np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0],
                                        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]], dtype=float),
        }

    def draw_structures(structures, titles):
        """Display ASE structures in WEAS, with matching initial camera settings."""
        centered = [coords - coords.mean(axis=0) for coords in structures]
        extent = max(1.0, max(float(np.linalg.norm(coords, axis=1).max()) for coords in centered))
        panels = []
        for coords, title in zip(centered, titles):
            atoms = Atoms("Ar" * len(coords), positions=coords, pbc=False)
            # WeasWidget wraps its AnyWidget canvas in an ipywidgets HBox.
            # Pass the canvas to marimo, rather than the unsupported HBox.
            viewer = WeasWidget(
                from_ase=atoms,
                viewerStyle={"width": "100%", "height": "360px"},
                modelStyle=1,
                atomLabelType="Index",
                cameraSetting={"lookAt": [0, 0, 0], "direction": [1, 1, 0.8],
                               "distance": 4 * (extent + 2), "zoom": 1},
                cellSettings={"showCell": False},
            )
            viewer.avr.bond.add_bond_pair("Ar", max=1.35 * sigma, color1="#888888", color2="#888888")
            panels.append(mo.vstack([mo.md(f"**{title}**"), mo.ui.anywidget(viewer.children[0])]))
        return mo.vstack([
            mo.hstack(panels, widths="equal", align="start"),
            mo.md("Drag to rotate; scroll to zoom. Labels use **zero-based atom indices**. "
                  "Grey sticks mark pairs within 1.35σ for display only; all pairs contribute to the energy. "
                  "Viewer edits do not change the calculated results; rerun the cell to restore the computed structures."),
        ])

    return four_atom_shapes, draw_structures


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.4: a four-atom shape competition

    The scan varies **a**: nearest-neighbour spacing for the line, or edge
    length for the square and tetrahedron. All unique pairs contribute.
    Compare the lowest sampled energy of each shape; the static geometry
    plots show the corresponding structures. These are fixed-shape scans,
    not unrestricted optimizations.
    """)
    return


@app.cell(hide_code=True)
def _(benchmark_ok, calculate_LJ_cluster, four_atom_shapes, mo, np, plt):
    mo.stop(not benchmark_ok, mo.md("Fix the three-atom benchmark before comparing shapes."))

    def compare_shapes():
        lengths = np.linspace(3.2, 5.0, 361)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        geometry = plt.figure(figsize=(9, 3.2))
        rows = []
        for panel, name in enumerate(four_atom_shapes(1.0), start=1):
            energies = np.array([calculate_LJ_cluster(four_atom_shapes(length)[name]) for length in lengths])
            best = int(np.argmin(energies))
            minimum = float(np.min(energies))
            spacing = float(lengths[best])
            curve, = ax.plot(lengths, energies, label=name)
            ax.plot(spacing, minimum, "o", color=curve.get_color())
            rows.append({"Shape": name, "Best sampled a (Å)": spacing,
                         "Lowest sampled energy (eV)": minimum})

            coords = four_atom_shapes(spacing)[name]
            coords = coords - coords.mean(axis=0)
            view = geometry.add_subplot(1, 3, panel, projection="3d")
            view.scatter(*coords.T, s=130, color=curve.get_color(), depthshade=False)
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    if np.isclose(np.linalg.norm(coords[i] - coords[j]), spacing):
                        view.plot(*coords[[i, j]].T, color="grey", lw=1.5)
            radius = max(float(np.abs(coords).max()), spacing / 2) * 1.25
            view.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius),
                     title=f"{name}\na = {spacing:.3f} Å")
            view.set_box_aspect((1, 1, 1))
            view.view_init(elev=22, azim=-65)
            view.set_axis_off()
        ax.set(xlabel="Spacing / edge length a (Å)", ylabel="Total energy (eV)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        geometry.subplots_adjust(left=.01, right=.99, bottom=.02, top=.78, wspace=.05)
        return mo.vstack([
            mo.ui.table(rows, selection=None), fig, geometry,
            mo.md("Sampled minima on a **0.005 Å** grid. Structure panels are scaled separately; "
                  "grey lines show nearest neighbours, but all pairs enter the energy."),
        ])

    compare_shapes()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.5: let an optimizer move the atoms

    SciPy's **L-BFGS-B** method repeatedly changes coordinates to try to lower
    your cluster energy. It estimates derivatives numerically; you do not
    need to implement a gradient. We optimize in units of σ and ε for sensible
    numerical scales, then report Å and eV.

    The seed selects a reproducible random starting structure with no very
    close pairs. Submit the controls to run a new experiment. Start with
    **7 atoms**, then try seeds **1, 2, and 3**. Atom indices match between
    the initial and final interactive views; both start with the same camera
    settings. Rotate and zoom to inspect the structures.
    """)
    optimizer_controls = mo.ui.dictionary({
        "atoms": mo.ui.slider(4, 13, step=1, value=7, label="Number of atoms", show_value=True),
        "seed": mo.ui.number(0, 9999, step=1, value=1, label="Random seed"),
    }).form(submit_button_label="Run optimization", show_clear_button=False)
    optimizer_controls
    return (optimizer_controls,)


@app.cell(hide_code=True)
def _(calculate_LJ_cluster, epsilon, minimize, np, sigma):
    def random_cluster(n, seed):
        rng = np.random.default_rng(seed)
        points = []
        for attempt in range(10000):
            candidate = rng.uniform(-1.1, 1.1, size=3)
            if all(np.linalg.norm(candidate - point) > 0.95 for point in points):
                points.append(candidate)
            if len(points) == n:
                coords = sigma * np.array(points)
                return coords - coords.mean(axis=0)
        raise RuntimeError("Could not generate a separated starting cluster; try another seed.")

    def optimize_cluster(n, seed):
        initial = random_cluster(n, seed)

        def objective(flat_coords):
            return float(calculate_LJ_cluster(sigma * flat_coords.reshape(n, 3))) / epsilon

        history = [objective((initial / sigma).ravel()) * epsilon]

        def record_step(flat_coords):
            history.append(objective(flat_coords) * epsilon)

        result = minimize(objective, (initial / sigma).ravel(), method="L-BFGS-B",
                          callback=record_step,
                          options={"maxiter": 1200, "maxfun": 100000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 40})
        final = sigma * result.x.reshape(n, 3)
        return initial, final, result, np.array(history)

    return random_cluster, optimize_cluster


@app.cell(hide_code=True)
def _(benchmark_ok, mo, optimize_cluster, optimizer_controls):
    mo.stop(not benchmark_ok, mo.md("Fix the benchmark before running the optimizer."))
    mo.stop(optimizer_controls.value is None, mo.md("Select the atom count and seed, then press **Run optimization**."))
    optimization_settings = dict(optimizer_controls.value)
    initial_coords, final_coords, optimization_result, energy_history = optimize_cluster(
        int(optimization_settings["atoms"]), int(optimization_settings["seed"])
    )
    return optimization_settings, initial_coords, final_coords, optimization_result, energy_history


@app.cell(hide_code=True)
def _(calculate_LJ_cluster, draw_structures, energy_history, final_coords, initial_coords, mo, np, optimization_result, optimization_settings, plt):
    def show_optimization():
        initial_energy = float(calculate_LJ_cluster(initial_coords))
        final_energy = float(calculate_LJ_cluster(final_coords))
        distances_initial = [np.linalg.norm(initial_coords[i] - initial_coords[j])
                             for i in range(len(initial_coords)) for j in range(i + 1, len(initial_coords))]
        distances_final = [np.linalg.norm(final_coords[i] - final_coords[j])
                           for i in range(len(final_coords)) for j in range(i + 1, len(final_coords))]
        summary = [
            {"State": "Initial", "Energy (eV)": initial_energy, "Closest pair (Å)": float(min(distances_initial))},
            {"State": "Final", "Energy (eV)": final_energy, "Closest pair (Å)": float(min(distances_final))},
        ]
        coordinate_rows = [
            {"Atom index": i, "Initial x (Å)": float(start[0]), "Initial y (Å)": float(start[1]),
             "Initial z (Å)": float(start[2]), "Final x (Å)": float(end[0]),
             "Final y (Å)": float(end[1]), "Final z (Å)": float(end[2])}
            for i, (start, end) in enumerate(zip(initial_coords, final_coords))
        ]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(np.arange(len(energy_history)), energy_history)
        ax.set(xlabel="Accepted optimizer iteration (0 = initial)", ylabel="Total energy (eV)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return mo.vstack([
            mo.md(f"**N = {optimization_settings['atoms']}; seed = {optimization_settings['seed']}**"),
            mo.ui.table(summary, selection=None),
            draw_structures([initial_coords, final_coords],
                            [f"Initial: {initial_energy:.6f} eV", f"Final: {final_energy:.6f} eV"]),
            mo.md(f"**Solver success:** {bool(optimization_result.success)}. **Status:** {optimization_result.message}\n\n"
                  f"**Iterations:** {optimization_result.nit}; **function evaluations:** {optimization_result.nfev}; "
                  f"**max absolute numerical gradient (reduced units):** {np.max(np.abs(optimization_result.jac)):.3g}."),
            mo.callout("Energy increased: inspect the implementation and solver status.", kind="warn")
            if final_energy > initial_energy else mo.md("Final energy is no higher than initial energy."),
            fig,
            mo.md("**Initial and final coordinates** (unshifted data, Å). Viewers are centred only for display; atom indices start at zero."),
            mo.ui.table(coordinate_rows, selection=None),
            mo.md("**Interpret:** did different seeds reach the same energy? Solver success means a stopping criterion was met; "
                  "it does not prove a global minimum or even certify a local minimum. Rotated structures can have the same energy. "
                  "Report what happened, including failures or repeated outcomes."),
        ])

    show_optimization()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.6: how could this be faster?

    The optimizer calls your cluster function many times, and each call
    loops over every unique pair. Suggest one way to reduce the computation
    time without changing the Lennard-Jones model. No new implementation is required.
    """)
    return


if __name__ == "__main__":
    app.run()
