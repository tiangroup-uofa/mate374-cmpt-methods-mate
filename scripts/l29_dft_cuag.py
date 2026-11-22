"""L29 reference data: Cu, Ag, and ordered Cu-Ag cells with GPAW (PBE), plus atomic orbitals.

Needs a working GPAW installation with PBE setups (not part of the course environment).
Each structure runs in its own process. Results go to data/l29/:

  cuag_pbe.json       energy per atom, volume, lattice constant for each cell
  cuag_l10_density.npz valence (pseudo) density on the x = 0 plane of the CuAg L1_0 cell
  atom_orbitals.json  all-electron PBE radial densities and eigenvalues of Cu and Ag

Run from the project root, for example:

  PYTHONNOUSERSITE=1 python scripts/l29_dft_cuag.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "l29"
HARTREE = 27.211386
BOHR = 0.529177
# Ag sites in the 4-atom conventional fcc cell, and a starting lattice constant (Å)
CELLS = {
    "Cu": ([], 3.63),
    "Cu3Ag": ([0], 3.75),
    "CuAg": ("z_half", 3.90),
    "CuAg3": ([1, 2, 3], 4.03),
    "Ag": ([0, 1, 2, 3], 4.15),
}
SETTINGS = dict(ecut=450, kpts=(10, 10, 10), smearing=0.1)


def build(label, a):
    from ase.build import bulk
    at = bulk("Cu", "fcc", a=a, cubic=True)
    sites = CELLS[label][0]
    if sites == "z_half":
        sites = [i for i, p in enumerate(at.get_scaled_positions()) if abs(p[2] - 0.5) < 1e-6]
    syms = ["Cu"] * 4
    for i in sites:
        syms[i] = "Ag"
    at.set_chemical_symbols(syms)
    return at


def calculator(txt):
    from gpaw import GPAW, PW, FermiDirac
    return GPAW(mode=PW(SETTINGS["ecut"]), xc="PBE", kpts=SETTINGS["kpts"],
                occupations=FermiDirac(SETTINGS["smearing"]), txt=txt)


def run_cell(label):
    from ase.eos import EquationOfState
    a0 = CELLS[label][1]
    vols, ens = [], []
    for s in np.linspace(0.97, 1.03, 5):
        at = build(label, a0 * s)
        at.calc = calculator(None)
        ens.append(at.get_potential_energy())
        vols.append(at.get_volume())
    v0, e0, _ = EquationOfState(vols, ens).fit()
    at = build(label, 1.0)
    x_ag = at.get_chemical_symbols().count("Ag") / 4
    result = {"label": label, "x_Ag": x_ag, "E_atom": e0 / 4, "V_atom": v0 / 4, "a": v0 ** (1 / 3)}
    if label == "CuAg":
        at = build(label, v0 ** (1 / 3))
        at.calc = calculator(None)
        at.get_potential_energy()
        n = at.calc.get_pseudo_density()          # electrons / Å^3 on the real-space grid
        # keep only the x = 0 plane, indexed (y, z); it holds atoms at (0,0,0) and (0,1/2,1/2)
        np.savez_compressed(OUT / "cuag_l10_density.npz", density=n[0], a=v0 ** (1 / 3),
                            symbols=np.array(at.get_chemical_symbols()),
                            scaled=at.get_scaled_positions())
    print(json.dumps(result))


def run_atoms():
    from gpaw.atom.aeatom import AllElectronAtom
    out = {}
    for sym, shells in (("Cu", [(3, 2), (4, 0)]), ("Ag", [(4, 2), (5, 0)])):
        atom = AllElectronAtom(sym, xc="PBE", log=None)
        atom.run()
        r = atom.rgd.r_g
        d = {"r_angstrom": (r * BOHR).tolist()}
        for n, l in shells:
            ch = atom.channels[l]
            i = n - l - 1
            p = (r * ch.phi_ng[i]) ** 2
            p /= np.trapz(p, r * BOHR)
            d[f"{n}{'spd'[l]}"] = {"e_eV": float(ch.e_n[i] * HARTREE), "P": p.tolist()}
        out[sym] = d
    (OUT / "atom_orbitals.json").write_text(json.dumps(out))


def run_dos():
    """Density of states of fcc Cu and Ag, and the SCF history of the CuAg cell."""
    out = {}
    for sym, a in (("Cu", 3.644), ("Ag", 4.146)):
        from ase.build import bulk
        at = bulk(sym, "fcc", a=a)
        at.calc = calculator(None)
        at.get_potential_energy()
        e, dos = at.calc.get_dos(spin=0, npts=600, width=0.15)
        out[sym] = {"E_minus_EF": (e - at.calc.get_fermi_level()).tolist(), "dos": dos.tolist()}
    (OUT / "dos_cu_ag.json").write_text(json.dumps(out))
    log = OUT / "cuag_scf.txt"
    at = build("CuAg", 3.918)
    at.calc = calculator(str(log))
    at.get_potential_energy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "dos":
        run_dos()
        return
    if len(sys.argv) > 1:
        run_cell(sys.argv[1])
        return
    run_atoms()
    procs = {k: subprocess.Popen([sys.executable, __file__, k], stdout=subprocess.PIPE, text=True)
             for k in CELLS}
    results = {k: json.loads(p.communicate()[0].strip().splitlines()[-1]) for k, p in procs.items()}
    cu, ag = results["Cu"]["E_atom"], results["Ag"]["E_atom"]
    for r in results.values():
        x = r["x_Ag"]
        r["dH_meV"] = 1000 * (r["E_atom"] - ((1 - x) * cu + x * ag))
    (OUT / "cuag_pbe.json").write_text(json.dumps({"settings": SETTINGS, "cells": results}, indent=1))
    for r in results.values():
        print(f"{r['label']:6s} x_Ag={r['x_Ag']:.2f} a={r['a']:.3f} Å  dH={r['dH_meV']:+.1f} meV/atom")


if __name__ == "__main__":
    main()
