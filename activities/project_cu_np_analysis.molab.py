# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ase",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "weas-widget",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Reading and plotting your Cu nanoparticle runs

    This notebook shows how to read the files from the simulation notebook and how to draw them.
    Each cell is a short example. Copy and change the code to make the figures for your report.

    - `.csv` files hold the energies and temperatures, one row per record.
    - `.xyz` files hold the atom positions at a few moments of the run.

    Upload your files below, or put them in a `runs/` folder next to this notebook.
    """)
    return


@app.cell
def imports():
    import io
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read
    from scipy.spatial import cKDTree
    from scipy.special import sph_harm_y

    KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K
    RUN_DIR = Path(mo.notebook_dir() or Path.cwd()) / "runs"
    return KB_EV, RUN_DIR, cKDTree, io, mo, np, plt, read, sph_harm_y


@app.cell
def upload(mo):
    uploads = mo.ui.file(filetypes=[".csv", ".xyz"], multiple=True, kind="area",
                         label="Drop .csv and .xyz files here")
    uploads
    return (uploads,)


@app.cell
def files(RUN_DIR, mo, uploads):
    # Collect every file as {name: text}. An uploaded file replaces a file in runs/ with the same name.
    files = {p.name: p.read_text() for p in sorted(RUN_DIR.glob("*.*"))} if RUN_DIR.exists() else {}
    files.update({f.name: f.contents.decode() for f in uploads.value})
    csv_names = sorted(n for n in files if n.endswith(".csv"))
    xyz_names = sorted(n for n in files if n.endswith(".xyz"))
    mo.md(f"**Found:** {len(csv_names)} `.csv` and {len(xyz_names)} `.xyz` files.")
    return csv_names, files, xyz_names


@app.cell(hide_code=True)
def read_md(mo):
    mo.md(r"""
    ## 1 · Read a `.csv` file

    `np.genfromtxt` reads the table. With `names=True`, each column is found by the name in the first row,
    for example `data["time_ps"]`. The `stage` column holds text, so `dtype=None` keeps it as text.
    """)
    return


@app.cell
def read_csv(io, np):
    def load_run(text):
        """Read one .csv file into a table whose columns are found by name."""
        return np.genfromtxt(io.StringIO(text), delimiter=",", names=True, dtype=None, encoding="utf-8")
    return (load_run,)


@app.cell(hide_code=True)
def stability_md(mo):
    mo.md(r"""
    ## 2 · Energy against time (Experiment 1)

    Plot both energies per atom against time. The dashed line is $\tfrac32 k_\mathrm{B}T$ at the set temperature.
    """)
    return


@app.cell
def stability_pick(csv_names, mo):
    _stability = [n for n in csv_names if n.startswith("stability")]
    stability_file = mo.ui.dropdown(_stability, value=_stability[0] if _stability else None, label="File")
    stability_file
    return (stability_file,)


@app.cell
def stability_plot(KB_EV, files, load_run, mo, plt, stability_file):
    mo.stop(stability_file.value is None, mo.md("Choose a `stability_*.csv` file."))
    data = load_run(files[stability_file.value])

    fig1, (ax_pe, ax_ke) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax_pe.plot(data["time_ps"], data["PE_eV_per_atom"], lw=1, color="tab:red")
    ax_pe.set(xlabel="Time (ps)", ylabel="Potential energy (eV/atom)")
    ax_ke.plot(data["time_ps"], data["KE_eV_per_atom"], lw=1, color="tab:blue")
    ax_ke.axhline(1.5 * KB_EV * data["T_set_K"][0], color="k", ls="--", lw=1)
    ax_ke.set(xlabel="Time (ps)", ylabel="Kinetic energy (eV/atom)")
    fig1.tight_layout()
    fig1
    return


@app.cell(hide_code=True)
def cycle_md(mo):
    mo.md(r"""
    ## 3 · Energy against temperature (Experiment 2)

    Select rows by their `stage` to draw heating and cooling in different colours. Choose several files to put
    runs on the same axes, for example two heating rates or two shapes from your group.

    The lower panel shows the slope $\mathrm dE/\mathrm dT$ of the heating curve, smoothed over 21 K.
    The melting region appears as a peak, because the particle takes up its latent heat there.
    """)
    return


@app.cell
def cycle_pick(csv_names, mo):
    _cycles = [n for n in csv_names if n.startswith("cycle")]
    cycle_files = mo.ui.multiselect(_cycles, value=_cycles[:1], label="Files")
    cycle_files
    return (cycle_files,)


@app.cell
def cycle_plot(cycle_files, files, load_run, mo, np, plt):
    mo.stop(not cycle_files.value, mo.md("Choose one or more `cycle_*.csv` files."))

    def smooth(y, n=21):
        """Running mean over n points."""
        return np.convolve(y, np.ones(n) / n, mode="same")

    fig2, (ax_e, ax_slope) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=(2, 1))
    for i, name in enumerate(cycle_files.value):
        run = load_run(files[name])
        heat = run["stage"] == "heat"
        cool = run["stage"] == "cool"
        label = name.removeprefix("cycle_").removesuffix(".csv")
        ax_e.plot(run["T_set_K"][heat], run["PE_eV_per_atom"][heat], color=f"C{i}", lw=1, label=f"{label}, heating")
        ax_e.plot(run["T_set_K"][cool], run["PE_eV_per_atom"][cool], color=f"C{i}", lw=1, ls=":",
                  label=f"{label}, cooling")
        slope = np.gradient(smooth(run["PE_eV_per_atom"][heat]), run["T_set_K"][heat]) * 1000
        inside = slice(11, -11)                     # the running mean is incomplete near both ends
        ax_slope.plot(run["T_set_K"][heat][inside], slope[inside], color=f"C{i}", lw=1)
    ax_e.set(ylabel="Potential energy (eV/atom)")
    ax_e.legend(frameon=False, fontsize=8)
    ax_slope.set(xlabel="Thermostat temperature (K)", ylabel="dE/dT on heating\n(meV/atom/K)")
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def structure_md(mo):
    mo.md(r"""
    ## 4 · Look at the atoms

    `ase.io.read(..., index=":")` reads every frame of an `.xyz` file as a list. Each frame keeps the time,
    stage, and set temperature of the run in `frame.info`.

    - **3D viewer (WEAS):** rotate with the mouse and play the frames with the slider below the picture.
    - **Slice:** a flat picture of the atoms within 2 Å of the centre plane, which shows whether the inside of the
      particle is still a crystal. The slice works without a 3D viewer and goes straight into a report.
    """)
    return


@app.cell
def structure_pick(mo, xyz_names):
    _runs = [n for n in xyz_names if n.startswith("cycle")] or xyz_names
    structure_file = mo.ui.dropdown(xyz_names, value=_runs[0] if _runs else None, label="File")
    structure_file
    return (structure_file,)


@app.cell
def structure_read(files, io, mo, read, structure_file):
    mo.stop(structure_file.value is None, mo.md("Choose an `.xyz` file."))
    frames = read(io.StringIO(files[structure_file.value]), index=":", format="extxyz")
    frame_slider = mo.ui.slider(0, len(frames) - 1, value=len(frames) - 1, label="Frame for the slice",
                                show_value=True)
    frame_slider
    return frame_slider, frames


@app.cell
def viewer(frames, mo):
    from weas_widget import WeasWidget

    GUI = {  # keep the picture, the frame slider, and fullscreen; hide the extra panels
        "controls": {"enabled": False},
        "meshLegend": {"enabled": False},
        "atomLegend": {"enabled": False},
        "buttons": {"enabled": True, "fullscreen": True, "undo": False, "redo": False, "export": False,
                    "import": False, "measurement": False},
        "timeline": {"enabled": True},
    }
    viewer3d = WeasWidget(viewerStyle={"width": "420px", "height": "320px"}, guiConfig=GUI)
    viewer3d.from_ase(frames)
    mo.ui.anywidget(viewer3d._widget)
    return


@app.cell
def slice_plot(frame_slider, frames, np, plt):
    frame = frames[frame_slider.value]
    xyz = frame.positions - frame.positions.mean(axis=0)
    near_centre = np.abs(xyz[:, 2]) < 2.0          # atoms within 2 Angstrom of the z = 0 plane

    fig3, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.scatter(xyz[near_centre, 0], xyz[near_centre, 1], s=10, color="peru")
    ax.set_aspect("equal")
    ax.set(xlabel="x (Å)", ylabel="y (Å)",
           title=f"{frame.info.get('stage', '')}, {frame.info.get('T_set_K', '')} K, {frame.info.get('time_ps', '')} ps")
    fig3.tight_layout()
    fig3
    return


@app.cell(hide_code=True)
def gr_md(mo):
    mo.md(r"""
    ## 5 · Pair distribution function $g(r)$

    $g(r)$ is the number of atoms found at distance $r$ from an atom, divided by the number expected at the
    same average density with no structure.

    - **Crystal:** sharp peaks at the fcc neighbour distances, 2.56, 3.62, 4.43, and 5.11 Å.
    - **Liquid:** a broad first peak, then weaker and smoother oscillations.

    Only atoms at least 10 Å inside the surface are used as centres, so that every shell up to 8 Å is full.
    The plot compares frame 1 (the crystal at the start of the run) with the frame chosen by the slider above.
    """)
    return


@app.cell
def gr_function(cKDTree, np):
    def pair_distribution(positions, r_max=8.0, dr=0.05, inset=10.0):
        """g(r) of a free particle, averaged over atoms at least `inset` Angstrom inside the surface."""
        xyz = positions - positions.mean(axis=0)
        radius = np.linalg.norm(xyz, axis=1)
        centres = xyz[radius < radius.max() - inset]
        inner = radius.max() - 3.0                                  # density from the inside of the particle
        density = (radius < inner).sum() / (4 / 3 * np.pi * inner**3)
        distances = cKDTree(centres).sparse_distance_matrix(cKDTree(xyz), r_max, output_type="ndarray")["v"]
        edges = np.arange(0.0, r_max + dr, dr)
        counts, _ = np.histogram(distances[distances > 0], bins=edges)
        shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        return 0.5 * (edges[1:] + edges[:-1]), counts / (len(centres) * density * shell)
    return (pair_distribution,)


@app.cell
def gr_plot(frame_slider, frames, mo, np, pair_distribution, plt):
    _reference_index = min(1, len(frames) - 1)
    _selected = (frames[_reference_index], frames[frame_slider.value])
    _enough_core = all(np.max(np.linalg.norm(f.positions - f.positions.mean(axis=0), axis=1)) > 10.0
                       for f in _selected)
    mo.stop(not _enough_core, mo.md("This particle has no atoms 10 Å inside its outer radius. Choose a larger particle for this g(r) estimate."))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _index, _colour in ((_reference_index, "grey"), (frame_slider.value, "tab:red")):
        _r, _g = pair_distribution(frames[_index].positions)
        _info = frames[_index].info
        _ax.plot(_r, _g, color=_colour, lw=1,
                 label=f"frame {_index}: {_info.get('stage', '')}, {_info.get('T_set_K', '')} K")
    _ax.set(xlabel="r (Å)", ylabel="g(r)", xlim=(2, 8))
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def q6_md(mo):
    mo.md(r"""
    ## 6 · Crystal or liquid, atom by atom

    The bond-orientational order parameter $q_6$ describes the directions of the bonds around one atom
    (Steinhardt, Nelson, and Ronchetti, *Phys. Rev. B* **28**, 784, 1983). In a crystal, two neighbouring atoms
    see the same pattern of bond directions. In a liquid, they do not. The function below labels an atom
    **solid-like** when its pattern agrees with at least 6 of its neighbours (ten Wolde, Ruiz-Montero, and
    Frenkel, *J. Chem. Phys.* **104**, 9932, 1996).

    - **Slice:** the atoms near the centre plane, gold if solid-like and blue if liquid-like.
    - **Radial profile:** the solid-like fraction against the distance from the centre of the particle.
    """)
    return


@app.cell
def q6_function(cKDTree, np, sph_harm_y):
    def solid_like(positions, r_cut=3.05, agreement=0.7, bonds_min=6):
        """True for each atom whose q6 bond pattern agrees with at least `bonds_min` neighbours."""
        n = len(positions)
        pairs = cKDTree(positions).query_pairs(r_cut, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        bond = positions[j] - positions[i]
        polar = np.arccos(np.clip(bond[:, 2] / np.linalg.norm(bond, axis=1), -1, 1))
        azimuth = np.arctan2(bond[:, 1], bond[:, 0])
        ylm = np.stack([sph_harm_y(6, m, polar, azimuth) for m in range(-6, 7)], axis=1)
        q6 = np.zeros((n, 13), complex)
        np.add.at(q6, i, ylm)
        np.add.at(q6, j, ylm)                      # a bond points the opposite way from j, same Y_6m
        q6 /= np.maximum(np.linalg.norm(q6, axis=1, keepdims=True), 1e-12)
        agree = np.real(np.sum(q6[i] * np.conj(q6[j]), axis=1)) > agreement
        bonds = np.bincount(i[agree], minlength=n) + np.bincount(j[agree], minlength=n)
        return bonds >= bonds_min
    return (solid_like,)


@app.cell
def q6_plot(frame_slider, frames, np, plt, solid_like):
    _frame = frames[frame_slider.value]
    _xyz = _frame.positions - _frame.positions.mean(axis=0)
    _solid = solid_like(_xyz)
    _near = np.abs(_xyz[:, 2]) < 2.0
    _radius = np.linalg.norm(_xyz, axis=1)
    _edges = np.arange(0, _radius.max() + 2, 2.0)
    _which = np.digitize(_radius, _edges) - 1
    _profile = [(_solid[_which == k].mean() if (_which == k).any() else np.nan) for k in range(len(_edges) - 1)]

    _fig, (_ax_s, _ax_r) = plt.subplots(1, 2, figsize=(9, 3.6), width_ratios=(1, 1.4))
    _ax_s.scatter(*_xyz[_near & ~_solid, :2].T, s=10, color="tab:blue", label="liquid-like")
    _ax_s.scatter(*_xyz[_near & _solid, :2].T, s=10, color="goldenrod", label="solid-like")
    _ax_s.set_aspect("equal")
    _ax_s.set(xlabel="x (Å)", ylabel="y (Å)")
    _ax_s.legend(frameon=False, fontsize=8, loc="upper right")
    _ax_r.plot(0.5 * (_edges[1:] + _edges[:-1]), _profile, "o-", color="k", ms=4, lw=1)
    _ax_r.set(xlabel="Distance from the centre (Å)", ylabel="Solid-like fraction", ylim=(-0.05, 1.05))
    _fig.suptitle(f"{_frame.info.get('stage', '')}, {_frame.info.get('T_set_K', '')} K: "
                  f"{_solid.mean():.0%} solid-like", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def order_md(mo):
    mo.md(r"""
    ## 7 · Solid-like fraction through the whole run

    The same label, applied to every saved frame, gives an order parameter for the whole particle:
    about 1 for a crystal and about 0 for a liquid. The core (inner half of the radius) and the outer shell
    (the outer 5 Å) are also shown separately.
    """)
    return


@app.cell
def order_plot(frames, mo, np, plt, solid_like):
    _rows = []
    for _frame in mo.status.progress_bar(frames, title="Labelling every frame", remove_on_exit=True):
        _xyz = _frame.positions - _frame.positions.mean(axis=0)
        _radius = np.linalg.norm(_xyz, axis=1)
        _solid = solid_like(_xyz)
        _rows.append((_frame.info.get("stage", ""), float(_frame.info.get("T_set_K", np.nan)), _solid.mean(),
                      _solid[_radius < 0.5 * _radius.max()].mean(), _solid[_radius > _radius.max() - 5].mean()))
    _stage = np.array([r[0] for r in _rows])
    _values = np.array([r[1:] for r in _rows])

    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _name, _colour in (("heat", "tab:red"), ("cool", "tab:blue")):
        _k = _stage == _name
        _ax.plot(_values[_k, 0], _values[_k, 1], "o-", ms=3, lw=1, color=_colour, label=f"{_name}, whole particle")
        _ax.plot(_values[_k, 0], _values[_k, 2], "--", lw=1, color=_colour, label=f"{_name}, core")
        _ax.plot(_values[_k, 0], _values[_k, 3], ":", lw=1, color=_colour, label=f"{_name}, outer 5 Å")
    _ax.set(xlabel="Thermostat temperature (K)", ylabel="Solid-like fraction", ylim=(-0.05, 1.05))
    _ax.legend(frameon=False, fontsize=7, ncol=2)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
