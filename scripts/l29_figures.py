"""L29 figures from the GPAW reference data in data/l29 (see scripts/l29_dft_cuag.py).

    uv run python scripts/l29_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "l29"
ASSETS = ROOT / "assets"
CU, AG = "#c26a2b", "#5f6f80"


def orbitals():
    d = json.loads((DATA / "atom_orbitals.json").read_text())
    a_cu = 3.615 / np.sqrt(2)          # experimental Cu-Cu nearest-neighbour distance, Å
    a_ag = 4.086 / np.sqrt(2)
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for a, sym, shells, col, nn in ((ax[0], "Cu", ("3d", "4s"), CU, a_cu), (ax[1], "Ag", ("4d", "5s"), AG, a_ag)):
        r = np.array(d[sym]["r_angstrom"])
        for k, ls in zip(shells, ("-", "--")):
            p = np.array(d[sym][k]["P"])
            a.plot(r, p, ls, color=col, lw=1.8, label=f"{k}   ε = {d[sym][k]['e_eV']:.1f} eV")
        a.axvline(nn / 2, color="0.5", lw=0.8, ls=":")
        a.text(nn / 2 + 0.06, 1.05, "half the distance\nto a neighbour", fontsize=8, color="0.35")
        a.set(xlim=(0, 4), xlabel="distance from the nucleus, r (Å)", title=f"{sym} atom (PBE, non-relativistic)")
        a.legend(frameon=False, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("radial probability density (1/Å)")
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-orbitals.png", dpi=300)


def density():
    f = DATA / "cuag_l10_density.npz"
    if not f.exists():
        return
    z = np.load(f)
    n = z["density"]
    a = float(z["a"])
    plane = n.T                           # stored as the x = 0 plane, indexed (y, z)
    plane = np.tile(plane, (2, 2))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(plane, origin="lower", extent=(0, 2 * a, 0, 2 * a), cmap="magma",
                   vmax=np.percentile(plane, 99))
    sp, sy = z["scaled"], z["symbols"]
    for p, s in zip(sp, sy):
        if abs(p[0]) < 1e-6:
            for dy in (0, 1, 2):
                for dz in (0, 1, 2):
                    y, zz = (p[1] + dy) * a, (p[2] + dz) * a
                    if y <= 2 * a + 1e-6 and zz <= 2 * a + 1e-6:
                        ax.text(y, zz, s, color="white", ha="center", va="center", fontsize=9, weight="bold")
    ax.set(xlabel="y (Å)", ylabel="z (Å)", title="valence electron density, CuAg (L1$_0$)")
    fig.colorbar(im, ax=ax, label="electrons per Å$^3$")
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-cuag-density.png", dpi=300)


def mixing():
    f = DATA / "cuag_pbe.json"
    if not f.exists():
        return
    cells = json.loads(f.read_text())["cells"]
    pts = sorted((c["x_Ag"], c["dH_meV"]) for c in cells.values())
    x, h = map(np.array, zip(*pts))
    om = np.mean([hh / (xx * (1 - xx)) for xx, hh in pts if 0 < xx < 1])
    xs = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(xs, om * xs * (1 - xs), color="0.4", lw=1.2, label=f"Ω x(1 − x), Ω = {om / 1000:.2f} eV")
    ax.plot(x, h, "o", color=CU, ms=7, label="DFT (PBE), ordered cells")
    for xx, hh in pts:
        if 0 < xx < 1:
            ax.annotate(f"{hh:+.0f}", (xx, hh), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="Ag fraction, x", ylabel="ΔH$_{mix}$ (meV/atom)", ylim=(-10, 130))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-mixing-enthalpy.png", dpi=300)


if __name__ == "__main__":
    orbitals()
    density()
    mixing()
