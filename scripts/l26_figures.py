"""Draw the L26 figures on interatomic energy models.

    uv run --locked python scripts/l26_figures.py

Inputs: activities/Cu_mishin1.eam.alloy, data/l26/l26-heating-*.csv (from
scripts/l26_potential_runs.py), and the Fall 2025 slide PDF when it is present.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data" / "l26"
DPI = 300
INK, GREY = "#231f20", "#5f6670"
RED, BLUE, GREEN, PURPLE, ORANGE = "#c0392b", "#1f5fa8", "#1e8449", "#7d3c98", "#d35400"
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none"})


def save(fig, name, svg=False):
    exts = ("png", "svg") if svg else ("png",)
    for ext in exts:
        fig.savefig(ASSETS / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print("Saved", name)


# ---------------------------------------------------------------- condensation movie
def hard_disk_events(pos, vel, box, diameter, t_end, frame_dt):
    """Exact event-driven hard-disk dynamics in a periodic square box."""
    n = len(pos)
    frames, t, next_frame = [], 0.0, 0.0
    iu = np.triu_indices(n, 1)
    while next_frame <= t_end:
        dr = pos[iu[1]] - pos[iu[0]]
        dr -= box * np.round(dr / box)
        dv = vel[iu[1]] - vel[iu[0]]
        b = np.sum(dr * dv, axis=1)
        vv = np.sum(dv * dv, axis=1)
        disc = b * b - vv * (np.sum(dr * dr, axis=1) - diameter**2)
        ok = (b < 0) & (disc > 0)
        tc = np.full(len(b), np.inf)
        tc[ok] = (-b[ok] - np.sqrt(disc[ok])) / vv[ok]
        k = int(np.argmin(tc))
        dt = max(tc[k], 0.0)
        while next_frame <= t + dt and next_frame <= t_end:
            frames.append(np.mod(pos + vel * (next_frame - t), box))
            next_frame += frame_dt
        pos = pos + vel * dt
        t += dt
        i, j = iu[0][k], iu[1][k]
        rij = pos[j] - pos[i]
        rij -= box * np.round(rij / box)
        rij /= np.linalg.norm(rij)
        dvn = np.dot(vel[j] - vel[i], rij)
        vel[i] += dvn * rij
        vel[j] -= dvn * rij
    return frames


def lj_langevin(pos, vel, box, temp, t_end, frame_dt, dt=0.005, gamma=0.5, seed=2):
    """2D Lennard-Jones particles (cutoff 2.5) with a BAOAB Langevin thermostat."""
    rng = np.random.default_rng(seed)
    rc2 = 2.5**2

    def forces(x):
        dr = x[:, None, :] - x[None, :, :]
        dr -= box * np.round(dr / box)
        r2 = np.sum(dr * dr, axis=2)
        np.fill_diagonal(r2, np.inf)
        inv6 = np.where(r2 < rc2, 1.0 / r2**3, 0.0)
        coef = 24 * inv6 * (2 * inv6 - 1) / r2
        return np.sum(coef[:, :, None] * dr, axis=1)

    f = forces(pos)
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * temp)
    stride = int(round(frame_dt / dt))
    frames = []
    for step in range(int(t_end / dt) + 1):
        if step % stride == 0:
            frames.append(np.mod(pos, box).copy())
        vel += 0.5 * dt * f
        pos += 0.5 * dt * vel
        vel = c1 * vel + c2 * rng.standard_normal(vel.shape)
        pos += 0.5 * dt * vel
        f = forces(pos)
        vel += 0.5 * dt * f
    return frames


def condensation_movie():
    n, density, temp = 100, 0.10, 0.40
    box = np.sqrt(n / density)
    rng = np.random.default_rng(374)
    m = int(np.ceil(np.sqrt(n)))
    grid = (np.stack(np.meshgrid(np.arange(m), np.arange(m)), -1).reshape(-1, 2)[:n] + 0.5) * box / m
    vel = rng.normal(0, np.sqrt(temp), (n, 2))
    vel -= vel.mean(0)
    t_end, frame_dt = 150.0, 1.5
    hard = hard_disk_events(grid.copy(), vel.copy(), box, 1.0, t_end, frame_dt)
    soft = lj_langevin(grid.copy(), vel.copy(), box, temp, t_end, frame_dt)
    nf = min(len(hard), len(soft))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    titles = ["Hard disks: repulsion only", "Lennard–Jones: repulsion + attraction"]
    colours = [GREY, BLUE]
    patches = []
    for ax, title, colour in zip(axes, titles, colours):
        ax.set(xlim=(0, box), ylim=(0, box), aspect="equal", xticks=[], yticks=[])
        ax.set_title(title, fontsize=13)
        ps = [ax.add_patch(Circle((0, 0), 0.5, fc=colour, ec=INK, lw=0.4)) for _ in range(n)]
        patches.append(ps)
    label = fig.text(0.5, 0.03, "", ha="center", fontsize=12, color=GREY)

    def draw(k):
        for ps, frames in zip(patches, (hard, soft)):
            for p, xy in zip(ps, frames[k]):
                p.center = xy
        label.set_text(f"Same number density and temperature, kT = {temp:.2f} ε.   t = {k * frame_dt:5.1f} τ")
        return [p for ps in patches for p in ps] + [label]

    FuncAnimation(fig, draw, frames=nf, blit=False).save(ASSETS / "L26-condensation.gif",
                                                         writer=PillowWriter(fps=10), dpi=110)
    draw(nf - 1)
    save(fig, "L26-condensation-final")
    print("Saved L26-condensation.gif")


# ---------------------------------------------------------------- pair-curve properties
def pair_properties():
    r = np.linspace(0.93, 2.6, 800)
    u = 4 * (r**-12 - r**-6)
    f = 48 * r**-13 - 24 * r**-7  # F = -dU/dr, positive = repulsive
    r0, rmax = 2 ** (1 / 6), (26 / 7) ** (1 / 6)
    fmax = 48 * rmax**-13 - 24 * rmax**-7
    k = 4 * (156 * r0**-14 - 42 * r0**-8)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True, gridspec_kw={"hspace": 0.08})
    a1.plot(r, u, color=BLUE, lw=2.6)
    a1.axhline(0, color=GREY, lw=0.8)
    rr = np.linspace(r0 - 0.07, r0 + 0.07, 50)
    a1.plot(rr, -1 + 0.5 * k * (rr - r0) ** 2, color=ORANGE, lw=2, ls="--")
    a1.annotate("spring: curvature $k=U''(r_0)$", (r0 + 0.06, -1 + 0.5 * k * 0.06**2), (1.36, -1.13),
                color=ORANGE, va="center", arrowprops=dict(arrowstyle="->", color=ORANGE))
    a1.plot([r0, 2.0], [-1, -1], color=RED, ls=":", lw=1.0)
    a1.annotate("", (2.0, 0), (2.0, -1), arrowprops=dict(arrowstyle="<->", color=RED))
    a1.text(2.03, -0.45, r"bond energy $\varepsilon$", color=RED, va="center")
    a1.plot([rmax], [4 * (rmax**-12 - rmax**-6)], "o", color=PURPLE)
    a1.text(rmax + 0.04, -0.82, "inflection point", color=PURPLE)
    a1.set(ylim=(-1.2, 1.0), ylabel=r"$U(r)/\varepsilon$")
    a2.plot(r, f, color=GREEN, lw=2.6)
    a2.axhline(0, color=GREY, lw=0.8)
    a2.plot([rmax, rmax], [fmax, 0], color=PURPLE, ls=":", lw=1.2)
    a2.plot([rmax], [fmax], "o", color=PURPLE)
    a2.annotate(r"largest restoring force $F_0$ at $r_{\max}$", (rmax, fmax), (1.45, -1.8), color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    a2.text(1.75, 0.55, "repulsive", color=GREY)
    a2.text(1.75, -0.75, "attractive", color=GREY)
    a2.set(ylim=(-3.0, 4.0), xlabel=r"separation $r/\sigma$", ylabel=r"$F(r)=-\mathrm{d}U/\mathrm{d}r$")
    for ax in (a1, a2):
        ax.axvline(r0, color=INK, ls=":", lw=1.0)
        ax.spines[["top", "right"]].set_visible(False)
    a2.text(r0, -2.85, r"$r_0$", ha="center")
    a2.text(rmax, 0.25, r"$r_{\max}$", ha="center", color=PURPLE)
    a1.set_xlim(0.93, 2.6)
    save(fig, "L26-pair-properties", svg=True)


# ---------------------------------------------------------------- coordination
def coordination():
    z = np.array([4, 6, 8, 12])
    abinitio = np.array([2.293, 2.996, 3.496, 3.540])  # Mishin et al. (2001), Table VI, LDA
    names = ["diamond", "sc", "bcc", "fcc"]
    zz = np.linspace(0, 12.5, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(zz, 3.54 * zz / 12, color=RED, lw=2.2, label=r"pair model: $E\propto Z$")
    ax.plot(zz, 3.54 * np.sqrt(zz / 12), color=BLUE, lw=2.2, ls="--", label=r"many-body trend: $E\propto\sqrt{Z}$")
    ax.plot(z, abinitio, "o", color=INK, ms=8, label="Cu, ab initio (Mishin et al. 2001)")
    for zi, ei, nm in zip(z, abinitio, names):
        ax.annotate(nm, (zi, ei), (zi - 0.3, ei + 0.22), ha="right", fontsize=11)
    ax.set(xlim=(0, 12.8), ylim=(0, 4.2), xlabel="coordination number $Z$ (nearest neighbours)",
           ylabel="cohesive energy (eV/atom)")
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "L26-coordination-energy", svg=True)


# ---------------------------------------------------------------- EAM functions
def read_setfl(path):
    lines = path.read_text().splitlines()
    nrho, drho, nr, dr, cutoff = lines[4].split()
    nrho, nr, drho, dr, cutoff = int(nrho), int(nr), float(drho), float(dr), float(cutoff)
    vals = np.array(" ".join(lines[6:]).split(), dtype=float)
    embed, dens, rphi = vals[:nrho], vals[nrho:nrho + nr], vals[nrho + nr:nrho + 2 * nr]
    rho = np.arange(nrho) * drho
    r = np.arange(nr) * dr
    return rho, embed, r, dens, rphi, cutoff


def eam_functions():
    rho, embed, r, dens, rphi, cutoff = read_setfl(ROOT / "activities" / "Cu_mishin1.eam.alloy")
    m = (r > 1.6) & (r < 6.0)
    phi = np.where(r > 0, rphi / np.where(r > 0, r, 1), 0)
    a0 = 3.615
    shells = a0 * np.array([1 / np.sqrt(2), 1, np.sqrt(1.5), np.sqrt(2)])
    counts = np.array([12, 6, 24, 12])
    rho_bulk = np.sum(counts * np.interp(shells, r, dens))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    axes[0].plot(r[m], phi[m], color=RED, lw=2.4)
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].set(ylim=(-0.3, 1.0), xlabel="r (Å)", ylabel="eV", title=r"pair term $\phi(r)$")
    axes[1].plot(r[m], dens[m], color=GREEN, lw=2.4)
    axes[1].set(xlabel="r (Å)", title=r"density contribution $f(r)$")
    for s in shells:
        axes[0].axvline(s, color=GREY, ls=":", lw=0.8)
        axes[1].axvline(s, color=GREY, ls=":", lw=0.8)
    axes[1].text(shells[0] + 0.05, axes[1].get_ylim()[1] * 0.9, "fcc neighbour shells", color=GREY, fontsize=10)
    mm = rho < 1.6
    axes[2].plot(rho[mm], embed[mm], color=BLUE, lw=2.4)
    axes[2].axvline(rho_bulk, color=INK, ls=":", lw=1.2)
    axes[2].text(rho_bulk + 0.02, -0.4, "perfect fcc", fontsize=10)
    axes[2].set(xlabel=r"local density $\rho$", ylabel="eV", title=r"embedding energy $F(\rho)$")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "L26-eam-functions", svg=True)
    print("rho_bulk", rho_bulk, "cutoff", cutoff)


# ---------------------------------------------------------------- heating curves
MODELS = {"LJ": (RED, "Lennard–Jones (fitted to Cu)"), "EMT": (GREEN, "EMT (ASE)"),
          "EAM": (BLUE, "EAM (Mishin Cu)"), "MACE": (PURPLE, "MACE-MP-0 (ML)")}


def solid_like_fraction(positions, r_cut, agreement=0.7, bonds_min=6):
    from scipy.spatial import cKDTree
    from scipy.special import sph_harm_y
    count = len(positions)
    pairs = cKDTree(positions).query_pairs(r_cut, output_type="ndarray")
    if not len(pairs):
        return 1.0
    first, second = pairs[:, 0], pairs[:, 1]
    bonds = positions[second] - positions[first]
    lengths = np.linalg.norm(bonds, axis=1)
    polar = np.arccos(np.clip(bonds[:, 2] / lengths, -1, 1))
    azimuth = np.arctan2(bonds[:, 1], bonds[:, 0])
    harmonics = np.stack([sph_harm_y(6, m, polar, azimuth) for m in range(-6, 7)], axis=1)
    q = np.zeros((count, 13), complex)
    np.add.at(q, first, harmonics)
    np.add.at(q, second, harmonics)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    agrees = np.real(np.sum(q[first] * np.conj(q[second]), axis=1)) > agreement
    aligned = np.bincount(first[agrees], minlength=count) + np.bincount(second[agrees], minlength=count)
    return float(np.mean(aligned >= bonds_min))


def frame_q6(name):
    from ase.io import read
    frames = read(DATA / f"l26-frames-{name}.xyz", ":")
    r_cut = 2.556 * 1.19  # 1.19 x the fcc nearest-neighbour distance
    out = [(float(a.info["target_T_K"]), solid_like_fraction(a.positions, r_cut)) for a in frames]
    return np.array(out)


def heating():
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1, ax2 = axes
    for name, (colour, label) in MODELS.items():
        path = DATA / f"l26-heating-{name}.csv"
        if not path.exists():
            continue
        d = np.genfromtxt(path, delimiter=",", names=True)
        ramp = d["time_ps"] > 2.0
        t, e = d["target_T_K"][ramp], d["potential_energy_eV_per_atom"][ramp]
        k = 8
        es = np.convolve(e, np.ones(k) / k, mode="valid")
        ts = t[k // 2: k // 2 + len(es)]
        e300 = np.mean(d["potential_energy_eV_per_atom"][(d["time_ps"] > 1.0) & (d["time_ps"] <= 2.0)])
        ax1.plot(ts, es - e300, color=colour, lw=2.2)
        q6 = frame_q6(name)
        ax2.plot(q6[:, 0], q6[:, 1], color=colour, lw=2.2)
        print(name, "q6 at 300/900/1300/1500 K:",
              [float(np.interp(T, q6[:, 0], q6[:, 1])) for T in (300, 900, 1300, 1500)])
    handles = [Line2D([], [], color=c, lw=2.2) for c, _ in MODELS.values()]
    labels = [l for _, l in MODELS.values()]
    ax1.axvline(1358, color=INK, ls=":", lw=1.2)
    ax1.text(1400, 0.04, "bulk Cu: 1358 K", fontsize=10)
    ax1.set(xlim=(300, 4000), xlabel="thermostat temperature (K)",
            ylabel="potential energy rise (eV/atom)")
    ax1.legend(handles, labels, frameon=False, fontsize=10.5, loc="upper left")
    ax2.axvline(1358, color=INK, ls=":", lw=1.2)
    ax2.set(xlim=(300, 4000), xlabel="thermostat temperature (K)",
            ylim=(0, 1.05), ylabel="solid-like fraction ($q_6$ criterion)")
    ax2.legend(handles, labels, frameon=False, fontsize=10.5, loc="center right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "L26-heating-curves")


# ---------------------------------------------------------------- reused slides
def slides():
    pdf = ROOT / "instructor-private" / "fall-2025" / "lectures" / "MATE 374 Lecture 32.pdf"
    old = ROOT / "instructor-private" / "fall-2025" / "lectures" / "MATE 374 Lecture 28.pdf"
    for src, page, name in [(pdf, 8, "L26-zhang-s32-08"), (pdf, 11, "L26-zhang-s32-11"),
                            (pdf, 20, "L26-zhang-s32-20"), (old, 10, "L26-zhang-s28-10")]:
        if not src.exists():
            print("missing", src)
            continue
        stem = ASSETS / name
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", "-f", str(page), "-l", str(page),
                        str(src), str(stem)], check=True)
        print("Saved", name)


if __name__ == "__main__":
    import sys
    jobs = {"movie": condensation_movie, "pair": pair_properties, "coord": coordination,
            "eam": eam_functions, "heat": heating, "slides": slides}
    for key in (sys.argv[1:] or jobs):
        jobs[key]()
