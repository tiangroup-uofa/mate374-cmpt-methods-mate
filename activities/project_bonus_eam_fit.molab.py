# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.25.2", "numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2", "ase==3.29.0"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Bonus example · Fit an EAM-type potential

    Fit a copper EAM-type model to **VASP DFT energies and forces** for distorted fcc, bcc,
    hcp, and (111) slab structures. The calculations use PBE PAW, a 400 eV cutoff,
    a $4\times4\times4$ bulk mesh or $4\times4\times1$ slab mesh, and 0.15 eV smearing.
    Energy labels are the finite-smearing free energies consistent with the forces.

    Fit the model here, download its tables, and insert it into your own simulation notebook.
    That integration and the subsequent force and trajectory checks are your bonus task.
    The small-cell data cover distorted crystals and a surface. Compare the model with the Mishin
    reference, especially when you apply it to liquid copper or a nanoparticle.

    Launch locally with `uvx marimo@0.25.2 edit --sandbox project_bonus_eam_fit.molab.py`.
    The ZIP includes both data files in this folder. Keep the data files beside the notebook after extraction.
    """)
    return


@app.cell
def imports():
    import io
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.build import bulk
    from ase.io import read
    from ase.neighborlist import neighbor_list
    return Path, bulk, io, mo, neighbor_list, np, plt, read


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## Model and data

    The energy is a sum of pair interactions and an embedding energy for each atom:

    $$E=\sum_{i<j}\phi(r_{ij})+\sum_i F(\rho_i)+NE_0,\quad
    \rho_i=\sum_{j\ne i}f(r_{ij}).$$

    Here $f(r)=[\max(r_c-r,0)/r_c]^4$, $r_c=5.5$ Å,
    $\phi(r)=\sum_k a_k\max(t_k-r,0)^3$, and
    $F(\rho)=c_1\sqrt\rho+c_2\rho^{3/2}+c_3\rho^2$.
    The knots $t_k$ lie between 2.2 and 5.5 Å. $E_0$ is a constant per atom.
    All 16 coefficients are linear. Forces are the negative position derivatives of this same energy.
    Periodic neighbours, including periodic images, are counted.

    The notebook reads `cu_vasp_train.xyz` and `cu_vasp_test.xyz` beside this notebook in the ZIP,
    or from `project/data/` in the course repository. You can also upload both files below.
    There are 178 training and 46 test structures;
    each split contains all four configuration families.
    """)
    return


@app.cell
def files_ui(mo):
    data_upload = mo.ui.file(filetypes=[".xyz"], multiple=True, kind="area", label="Upload the training and test .xyz files")
    data_upload
    return (data_upload,)


@app.cell
def load_data(Path, data_upload, io, mo, read):
    names = ("cu_vasp_train.xyz", "cu_vasp_test.xyz")
    uploaded = {f.name: f.contents.decode("utf-8") for f in data_upload.value}
    notebook_dir = Path(mo.notebook_dir() or Path.cwd()).resolve()
    roots = (notebook_dir, notebook_dir.parent / "project" / "data")
    texts = {}
    for name in names:
        local_file = next((root / name for root in roots if (root / name).exists()), None)
        texts[name] = uploaded[name] if name in uploaded else (local_file.read_text() if local_file else None)
    missing = [name for name in names if texts[name] is None]
    mo.stop(bool(missing), mo.md("Upload both supplied files: " + ", ".join(missing)))
    train = read(io.StringIO(texts[names[0]]), index=":", format="extxyz")
    test = read(io.StringIO(texts[names[1]]), index=":", format="extxyz")
    mo.md(f"Loaded **{len(train)} training** and **{len(test)} test** structures.")
    return test, train


@app.cell(hide_code=True)
def design_md(mo):
    mo.md(r"""
    ## Build the linear system

    Each structure contributes 1 energy row and $3N$ force rows. We reserve about 20% of each
    configuration family in the training file for validation, using random seed 374. The separate test
    file is used after model selection. The training file is ordered by family, so taking its last 20%
    would validate only surface structures.

    Energy residuals are scaled by 0.03 eV/atom and force residuals by 0.05 eV/Å and $\sqrt{3N}$.
    These scales set the balance between energy and force errors. We also require relaxed fcc Cu to have
    lower energy than relaxed bcc and hcp Cu. Among the candidate fits that pass this crystal-ordering
    check, we select the lowest validation error and refit using the complete training file.
    """)
    return


@app.cell
def features(neighbor_list, np):
    knots = np.linspace(2.2, 5.5, 12)
    cutoff = 5.5

    def feature_rows(atoms):
        n = len(atoms)
        i, j, vectors, r = neighbor_list("ijDd", atoms, cutoff, self_interaction=False)
        direction = vectors / r[:, None]
        x = np.maximum(knots[None, :] - r[:, None], 0.0)
        pair = x**3
        pair_derivative = -3.0 * x**2
        density_pair = np.maximum(cutoff - r, 0.0)**4 / cutoff**4
        density_derivative = -4.0 * np.maximum(cutoff - r, 0.0)**3 / cutoff**4
        rho = np.bincount(i, weights=density_pair, minlength=n)
        root = np.sqrt(rho)
        embedding = np.column_stack((root, rho * root, rho**2))
        energy = np.r_[n, 0.5 * pair.sum(axis=0), embedding.sum(axis=0)]
        embed_derivative = np.column_stack((0.5 / np.maximum(root, 1e-12), 1.5 * root, 2 * rho))
        radial = np.column_stack((np.zeros(len(r)), 0.5 * pair_derivative,
                                  embed_derivative[i] * density_derivative[:, None]))
        contribution = direction[:, :, None] * radial[:, None, :]
        forces = np.zeros((n, 3, len(energy)))
        np.add.at(forces, i, contribution)
        np.add.at(forces, j, -contribution)
        return energy, forces

    def build_matrix(structures):
        energies, forces = zip(*(feature_rows(atoms) for atoms in structures))
        return np.array(energies), list(forces)
    return build_matrix, feature_rows


@app.cell
def matrices(build_matrix, mo, test, train):
    with mo.status.spinner(title="Building energy and force features"):
        energy_train, force_train = build_matrix(train)
        energy_test, force_test = build_matrix(test)
    mo.md(f"Design matrix ready: {energy_train.shape[1]} coefficients, including the atomic energy offset.")
    return energy_test, energy_train, force_test, force_train


@app.cell
def training_split(mo, np, train):
    split_rng = np.random.default_rng(374)
    fit_indices, validation_indices, split_counts = [], [], []
    families = sorted({a.info.get("family", "all") for a in train})
    for family in families:
        indices = np.array([i for i, a in enumerate(train) if a.info.get("family", "all") == family])
        split_rng.shuffle(indices)
        n_validation = max(1, round(0.2 * len(indices)))
        validation_indices.extend(indices[:n_validation])
        fit_indices.extend(indices[n_validation:])
        split_counts.append((family, len(indices) - n_validation, n_validation))
    fit_indices = np.array(sorted(fit_indices))
    validation_indices = np.array(sorted(validation_indices))
    split_table = "\n".join(f"| {name} | {n_fit} | {n_val} |" for name, n_fit, n_val in split_counts)
    mo.md("### Internal training and validation split\n\n| Family | Fit | Validation |\n|:--|--:|--:|\n" + split_table)
    return fit_indices, split_counts, validation_indices


@app.cell
def phase_features(bulk, feature_rows, np):
    # Energy per atom on a lattice-spacing grid. hcp uses ASE's ideal c/a ratio.
    phase_grids = {"fcc": np.linspace(3.2, 4.1, 121), "bcc": np.linspace(2.55, 3.25, 121),
                   "hcp": np.linspace(2.3, 2.9, 121)}
    phase_rows = {}
    for phase, spacings in phase_grids.items():
        phase_rows[phase] = np.array([feature_rows(atoms)[0] / len(atoms)
                                    for a in spacings for atoms in [bulk("Cu", phase, a=a)]])
    return phase_grids, phase_rows


@app.cell
def fitting(energy_test, energy_train, fit_indices, force_test, force_train, mo, np, phase_grids, phase_rows, test, train, validation_indices):
    ref_energy_train = np.array([a.info["dft_energy"] for a in train])
    ref_force_train = [a.arrays["dft_forces"] for a in train]
    ref_energy_test = np.array([a.info["dft_energy"] for a in test])
    ref_force_test = np.concatenate([a.arrays["dft_forces"].ravel() for a in test])
    test_sizes = np.array([len(a) for a in test])

    energy_scale = 0.03  # eV/atom
    force_scale = 0.05   # eV/Angstrom

    def weighted_matrix(energy, force, ref_e, ref_f):
        sizes = np.array([len(f) for f in force])
        a = np.vstack([energy / (sizes[:, None] * energy_scale),
                       *[f.reshape(-1, energy.shape[1]) / (force_scale * np.sqrt(3 * len(f))) for f in force]])
        b = np.concatenate([ref_e / (sizes * energy_scale),
                            *[f.ravel() / (force_scale * np.sqrt(3 * len(f))) for f in ref_f]])
        return a, b

    def fit_model(indices, penalty):
        a, b = weighted_matrix(energy_train[indices], [force_train[i] for i in indices],
                               ref_energy_train[indices], [ref_force_train[i] for i in indices])
        scale = np.linalg.norm(a, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        scaled = a / scale
        coeff = np.linalg.lstsq(np.vstack((scaled, np.sqrt(penalty) * np.eye(a.shape[1]))),
                                 np.r_[b, np.zeros(a.shape[1])], rcond=None)[0] / scale
        return coeff

    candidates = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    va, vb = weighted_matrix(energy_train[validation_indices], [force_train[i] for i in validation_indices],
                             ref_energy_train[validation_indices], [ref_force_train[i] for i in validation_indices])
    scores = [np.mean((va @ fit_model(fit_indices, alpha) - vb)**2) for alpha in candidates]
    full_fits = [fit_model(np.arange(len(train)), alpha) for alpha in candidates]
    phase_minima = [{phase: float((rows @ c).min()) for phase, rows in phase_rows.items()} for c in full_fits]
    phase_pass = [e["fcc"] < min(e["bcc"], e["hcp"]) for e in phase_minima]
    mo.stop(not any(phase_pass), mo.md("No candidate keeps fcc below both bcc and hcp. Revise the model or fitting scales before using it for Cu MD."))
    best_index = min((i for i, passes in enumerate(phase_pass) if passes), key=lambda i: scores[i])
    alpha_best = candidates[best_index]
    coefficients = full_fits[best_index]
    selected_phases = {phase: {"energy_eV_atom": phase_minima[best_index][phase],
                              "a_A": float(phase_grids[phase][np.argmin(rows @ coefficients)])}
                       for phase, rows in phase_rows.items()}
    predicted_e = energy_test @ coefficients / test_sizes
    ref_energy_test = ref_energy_test / test_sizes
    predicted_f = np.concatenate([(f @ coefficients).ravel() for f in force_test])
    e_mae = np.mean(np.abs(predicted_e - ref_energy_test)) * 1000
    f_mae = np.mean(np.abs(predicted_f - ref_force_test)) * 1000
    return alpha_best, candidates, coefficients, e_mae, energy_scale, f_mae, force_scale, phase_minima, phase_pass, predicted_e, predicted_f, ref_energy_test, ref_force_test, scores, selected_phases


@app.cell
def results(alpha_best, candidates, coefficients, e_mae, f_mae, mo, np, plt, predicted_e, predicted_f, ref_energy_test, ref_force_test, scores):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, reference, predicted, label in ((axes[0], ref_energy_test, predicted_e, "Energy (eV/atom)"),
                                             (axes[1], ref_force_test.ravel(), predicted_f.ravel(), "Force (eV/Å)")):
        ax.scatter(reference, predicted, s=8, alpha=0.4)
        lo, hi = min(reference.min(), predicted.min()), max(reference.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel=f"DFT {label}", ylabel=f"Model {label}")
    fig.tight_layout()
    mo.vstack([mo.md(f"**Held-out test:** energy MAE = {e_mae:.2f} meV/atom; force MAE = {f_mae:.1f} meV/Å.  \n"
                     f"Selected ridge penalty: {alpha_best:g}. Coefficients: `{np.array2string(coefficients, precision=4)}`"), fig])
    return


@app.cell
def force_check(candidates, coefficients, feature_rows, mo, np, phase_minima, phase_pass, scores, selected_phases, test):
    atom = test[0].copy()
    h = 1e-5  # Angstrom
    plus, minus = atom.copy(), atom.copy()
    plus.positions[0, 0] += h
    minus.positions[0, 0] -= h
    finite_difference = -(feature_rows(plus)[0] @ coefficients - feature_rows(minus)[0] @ coefficients) / (2 * h)
    analytic = feature_rows(atom)[1][0, 0] @ coefficients
    table = "\n".join(f"| {a:g} | {score:.5f} | {(phases['bcc'] - phases['fcc']) * 1000:.2f} | {'yes' if passes else 'no'} |"
                      for a, score, phases, passes in zip(candidates, scores, phase_minima, phase_pass))
    phase_table = "\n".join(f"| {name} | {values['a_A']:.3f} | {(values['energy_eV_atom'] - selected_phases['fcc']['energy_eV_atom']) * 1000:.2f} |"
                            for name, values in selected_phases.items())
    mo.md(f"""
    ## Check the force and penalty

    For atom 0 in the first test structure, the analytic x-force is **{analytic:.6f} eV/Å**.
    A central difference of the fitted energy gives **{finite_difference:.6f} eV/Å** with $h=10^{{-5}}$ Å.
    Their absolute difference is {abs(analytic - finite_difference):.2g} eV/Å.

    | Ridge penalty | Scaled validation MSE | bcc − fcc (meV/atom) | fcc lowest of 3? |
    |---:|---:|---:|:--|
    {table}

    The selected model gives these relaxed lattice energies. The grid varies the lattice spacing,
    with the ideal hcp $c/a$ ratio held fixed. This is a crystal-ordering check at 0 K.

    | Crystal | Lattice spacing (Å) | Energy above fcc (meV/atom) |
    |:--|---:|---:|
    {phase_table}
    """)
    return


@app.cell
def export_model(Path, alpha_best, coefficients, energy_scale, fit_indices, force_scale, io, mo, np, validation_indices):
    r_grid = np.linspace(0.0, 5.5, 10001)  # Angstrom
    rho_grid = np.linspace(0.0, 10.0, 20001)
    phi_table = np.maximum(np.linspace(2.2, 5.5, 12)[None, :] - r_grid[:, None], 0.0)**3 @ coefficients[1:13]
    density_table = np.maximum(5.5 - r_grid, 0.0)**4 / 5.5**4
    embed_table = coefficients[0] + np.column_stack((np.sqrt(rho_grid), rho_grid**1.5, rho_grid**2)) @ coefficients[13:]
    model_buffer = io.BytesIO()
    np.savez(model_buffer, phi=phi_table, density=density_table, embed=embed_table,
             r_max=5.5, rho_max=10.0, cutoff=5.5, coefficients=coefficients,
             element="Cu", length_unit="angstrom", energy_unit="eV", ridge_penalty=alpha_best,
             energy_scale=energy_scale, force_scale=force_scale, fit_indices=fit_indices, validation_indices=validation_indices)
    fitted_model_path = Path(mo.notebook_dir() or Path.cwd()) / "cu_fitted_eam.npz"
    fitted_model_path.write_bytes(model_buffer.getvalue())
    print(f"Fitted Cu model: {fitted_model_path}; energy MAE and force MAE appear above.")
    mo.vstack([mo.md("## Download your fitted model\nSaved beside this notebook as `cu_fitted_eam.npz`. "
                     "The NPZ file contains `phi`, `density`, `embed`, `r_max`, `rho_max`, and `cutoff`. "
                     "The embedding table includes the fitted constant energy per atom. "
                     "Check that every atom's density remains within the table range when you use it."),
               mo.download(model_buffer.getvalue(), filename="cu_fitted_eam.npz", label="Download fitted Cu model")])
    return (fitted_model_path,)


@app.cell(hide_code=True)
def tasks(mo):
    mo.md(r"""
    ## Investigate the fit

    1. Explain why energy and force rows are both linear in the coefficients.
    2. Compare the validation scores and crystal energies for the candidate ridge penalties. Which fits favour bcc over fcc? What happens when the penalty is too large?
    3. Change the number of pair knots or the relative energy/force scales. Refit and compare the held-out errors.
    4. Download the fitted tables above. Insert the fitted model into your copy of the Cu simulation notebook.
       Its `potential` cell reads EAM tables and `eam_system` builds the OpenMM force. Replace the
       reference tables with yours, retaining the Å-to-nm and eV-to-kJ/mol conversions.
    5. Check forces by finite differences and run a short trajectory before a complete heating cycle.
       Compare with Mishin using the same particle, seed, and rate. Report instability if it occurs.

    Report the data split, units, model, validation choice, test errors, and your finite-difference check.
    Save your edited notebook and include its link or source with the report.
    """)
    return


if __name__ == "__main__":
    app.run()
