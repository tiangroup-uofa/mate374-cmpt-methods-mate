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


def dimer():
    """Two atoms, one s electron each: bonding and antibonding levels for Cu-Cu, Ag-Ag, Cu-Ag."""
    t = {"Cu–Cu": 1.2, "Ag–Ag": 0.9, "Cu–Ag": 0.8}
    eps = {"Cu": 0.0, "Ag": 0.0}
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x0 = 0
    gains = {}
    for pair, tt in t.items():
        a, b = pair.split("–")
        ea, eb = eps[a], eps[b]
        mid, half = (ea + eb) / 2, np.hypot((ea - eb) / 2, tt)
        lo, hi = mid - half, mid + half
        gains[pair] = 2 * lo - (ea + eb)
        col_a = CU if a == "Cu" else AG
        col_b = CU if b == "Cu" else AG
        ax.plot([x0, x0 + 0.6], [ea, ea], color=col_a, lw=2.5)
        ax.plot([x0 + 2.4, x0 + 3.0], [eb, eb], color=col_b, lw=2.5)
        ax.plot([x0 + 1.2, x0 + 1.8], [lo, lo], color="k", lw=2.5)
        ax.plot([x0 + 1.2, x0 + 1.8], [hi, hi], color="k", lw=1.2, ls="--")
        for xa, ya in ((x0 + 0.6, ea), (x0 + 2.4, eb)):
            ax.plot([xa, x0 + 1.2 if xa < x0 + 1.5 else x0 + 1.8], [ya, lo], color="0.6", lw=0.7, ls=":")
            ax.plot([xa, x0 + 1.2 if xa < x0 + 1.5 else x0 + 1.8], [ya, hi], color="0.6", lw=0.7, ls=":")
        for dx, up in ((-0.1, True), (0.1, False)):
            y0a, y1a = (lo - 0.17, lo + 0.17) if up else (lo + 0.17, lo - 0.17)
            ax.annotate("", (x0 + 1.5 + dx, y1a), (x0 + 1.5 + dx, y0a),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=8))
        ax.text(x0 + 0.3, ea + 0.15, a, ha="center", fontsize=10, color=col_a)
        ax.text(x0 + 2.7, eb + 0.15, b, ha="center", fontsize=10, color=col_b)
        ax.text(x0 + 1.5, hi + 0.2, f"+{tt:g} eV", ha="center", fontsize=9)
        ax.text(x0 + 1.5, lo - 0.45, f"−{tt:g} eV", ha="center", fontsize=9)
        ax.text(x0 + 1.5, -2.3, f"{pair}\nbond energy {gains[pair]:+.1f} eV".replace("-", "−"), ha="center", fontsize=10)
        x0 += 3.8
    ax.set(xlim=(-0.4, x0 - 0.6), ylim=(-2.7, 1.8), ylabel="electron energy (eV)")
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    eps_mix = gains["Cu–Ag"] - 0.5 * (gains["Cu–Cu"] + gains["Ag–Ag"])
    ax.set_title(f"two electrons in the bonding level · ε = E(Cu–Ag) − ½[E(Cu–Cu) + E(Ag–Ag)] = {eps_mix:+.1f} eV",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-dimer-levels.png", dpi=300)


def ring_to_band():
    """Eigenvalues of pure rings of N atoms approach a continuous band from -2t to +2t."""
    t = 1.0
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    for k, N in enumerate((2, 4, 8, 16, 64)):
        if N == 2:
            levels = np.array([-t, t])                 # a pair: one bond, split ±t
        else:
            levels = -2 * t * np.cos(2 * np.pi * np.arange(N) / N)
        for i, e in enumerate(np.sort(levels)):
            filled = i < N // 2
            ax[0].plot([k - 0.32, k + 0.32], [e, e], color=CU if filled else "0.7", lw=1.4 if N < 64 else 0.6)
    ax[0].set_xticks(range(5), ["2", "4", "8", "16", "64"])
    ax[0].set(xlabel="atoms in the ring, N", ylabel="energy level / t", title="filled (coloured) and empty levels")
    ax[0].spines[["top", "right"]].set_visible(False)
    N = 2000
    lv = -2 * t * np.cos(2 * np.pi * np.arange(N) / N)
    ax[1].hist(lv, bins=60, density=True, orientation="horizontal", color="#d9b38c", edgecolor="#c26a2b", lw=0.4)
    e = np.linspace(-1.995, 1.995, 400)
    ax[1].plot(1 / (np.pi * np.sqrt(4 * t * t - e * e)), e, color="k", lw=1.2, label="exact, N → ∞")
    ax[1].axhline(0, color="0.4", lw=0.8, ls="--")
    ax[1].text(1.05, 0.08, "Fermi level", fontsize=8, color="0.3")
    ax[1].set(xlabel="density of levels", xlim=(0, 1.4), ylim=(-2.2, 2.2), title="N = 2000: a band")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-ring-to-band.png", dpi=300)


def dos():
    f = DATA / "dos_cu_ag.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for sym, col in (("Cu", CU), ("Ag", AG)):
        e, g = np.array(d[sym]["E_minus_EF"]), np.array(d[sym]["dos"])
        ax.plot(e, g, color=col, lw=1.6, label=f"fcc {sym}")
        ax.fill_between(e, g, where=e <= 0, color=col, alpha=0.18)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_ylim(0, 2.8)
    ax.text(0.15, 2.55, "Fermi level", fontsize=8)
    ax.annotate("Ag 4d band", (-4.6, 2.05), (-8.6, 2.45), fontsize=8, color=AG, arrowprops=dict(arrowstyle="->", color=AG, lw=0.7))
    ax.annotate("Cu 3d band", (-1.8, 2.3), (-3.6, 2.6), fontsize=8, color=CU, arrowprops=dict(arrowstyle="->", color=CU, lw=0.7))
    ax.text(1.0, 0.45, "s electrons:\nlow, flat", fontsize=8, color="0.3")
    ax.set(xlim=(-9, 4), xlabel="electron energy − Fermi level (eV)", ylabel="density of states (1/eV)",
           title="d band of Ag lies deeper than d band of Cu")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-dos.png", dpi=300)


def scf():
    import re
    f = DATA / "cuag_scf.txt"
    if not f.exists():
        return
    it, dens = [], []
    for line in f.read_text().splitlines():
        m = re.match(r"^iter:\s+(\d+)\s+\S+\s+(-?[\d.]+)\s+(?:c\s+)?(-?[\d.]+|\S+)\s+(?:c\s+)?(-?[\d.]+)", line)
        if m:
            it.append(int(m.group(1)))
            try:
                dens.append(float(m.group(4)))
            except ValueError:
                dens.append(np.nan)
    if not it:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(it, dens, "o-", color=CU, ms=4)
    ax.set(xlabel="iteration", ylabel="log$_{10}$ change in density", title="self-consistency, CuAg cell")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "L29-scf.png", dpi=300)


def chain():
    """Unit 6 in one picture: DFT energy -> lattice Monte Carlo -> composition field -> microstructure."""
    panels = [("L29-mixing-enthalpy.png", "DFT (L29)\nΔH$_{mix}$ ≈ +95 meV/atom"),
              ("L28-spinodal-snapshot.png", "Monte Carlo swaps (L28)\nε = Ω/4 in every swap"),
              ("L28-coarse-grain.png", "composition field (L28)\ng(c) with the same Ω"),
              ("L27-cuag-eutectic-sem.jpg", "Cu–Ag microstructure (L27)\nexperiment")]
    fig, ax = plt.subplots(1, 4, figsize=(11, 3.0))
    crops = {"L28-coarse-grain.png": lambda im: im[int(0.2 * im.shape[0]):int(0.92 * im.shape[0]), int(0.62 * im.shape[1]):int(0.89 * im.shape[1])],
             "L27-cuag-eutectic-sem.jpg": lambda im: im[: im.shape[0] // 2, : im.shape[1] // 2]}
    for a, (fn, lab) in zip(ax, panels):
        p = ASSETS / fn
        if p.exists():
            im = plt.imread(p)
            im = crops.get(fn, lambda x: x)(im)
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(lab, fontsize=9)
        a.axis("off")
    for k in range(3):
        fig.text(0.25 * (k + 1) - 0.005, 0.45, "→", fontsize=22, ha="center", va="center", color=CU)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(ASSETS / "L29-unit-chain.png", dpi=300)


if __name__ == "__main__":
    orbitals()
    density()
    mixing()
    dimer()
    ring_to_band()
    dos()
    scf()
    chain()
