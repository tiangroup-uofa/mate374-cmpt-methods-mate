"""Three regimes of the Cu-Ag lattice model for L28: nucleation, spinodal, mixed.

The swap rule is the pair-set Metropolis rule of the L28 GPU notebook, with the 16
pair sets in random order each sweep. Snapshots are saved to a cache so that the
figure can be redrawn without rerunning (about 15 minutes per case on one core):

    uv run python scripts/l28_regimes.py --cache /tmp/l28-regimes
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KTC = 1.0 / np.log(1.0 + np.sqrt(2.0))     # exact square lattice, k Tc / epsilon
SNAPS = (100, 1000, 10000, 100000)
CASES = [  # label, T/Tc, Ag fraction
    ("nucleation", 0.70, 0.03),
    ("spinodal", 0.70, 0.50),
    ("mixed", 1.30, 0.20),
]
T_SWEEP = 22e-6   # s per sweep, from tracer diffusion matched to Cu at 1000 K (see L28)


def pair_sets(N):
    y, x = np.mgrid[0:N, 0:N]
    f = lambda yy, xx: (yy % N) * N + xx % N
    out = []
    for axis in (0, 1):
        for a in range(4):
            for b in range(2):
                along, across = (x, y) if axis == 0 else (y, x)
                m = (along % 4 == a) & (across % 2 == b)
                py, px = y[m], x[m]
                dy, dx = (0, 1) if axis == 0 else (1, 0)
                ey, ex = dx, dy
                P, Q = f(py, px), f(py + dy, px + dx)
                nP = np.stack([f(py - dy, px - dx), f(py + ey, px + ex), f(py - ey, px - ex)])
                nQ = np.stack([f(py + 2 * dy, px + 2 * dx), f(py + dy + ey, px + dx + ex),
                               f(py + dy - ey, px + dx - ex)])
                out.append((P, Q, nP, nQ))
    return out


def run(t_over_tc, c0, N=256, seed=7, times=SNAPS):
    rng = np.random.default_rng(seed)
    kT = KTC * t_over_tc
    sets = pair_sets(N)
    s = np.zeros(N * N, np.int8)
    s[: round(c0 * N * N)] = 1
    rng.shuffle(s)
    snaps, t = {}, 0
    for target in times:
        while t < target:
            for k in rng.integers(16, size=16):
                P, Q, nP, nQ = sets[k]
                vp, vq, a, b = s[P], s[Q], s[nP], s[nQ]
                dE = (np.abs(a - vq).sum(0) + np.abs(b - vp).sum(0)
                      - np.abs(a - vp).sum(0) - np.abs(b - vq).sum(0))
                acc = (vp != vq) & ((dE <= 0) | (rng.random(P.size) < np.exp(-dE / kT)))
                s[P[acc]], s[Q[acc]] = vq[acc], vp[acc]
            t += 1
        snaps[str(target)] = s.reshape(N, N).copy()
    return snaps


def seconds(t):
    v = t * T_SWEEP
    if v < 1e-3:
        return f"{v * 1e6:.0f} µs"
    if v < 1e-2:
        return f"{v * 1e3:.1f} ms"
    return f"{v * 1e3:.0f} ms" if v < 1 else f"{v:.1f} s"


def coarse_grain(g, crop=64, block=8):
    """Lattice, block averages, and a smooth composition field from one snapshot."""
    s = g[:crop, :crop].astype(float)
    blocks = s.reshape(crop // block, block, crop // block, block).mean(axis=(1, 3))
    k = np.fft.fftfreq(crop)
    kx, ky = np.meshgrid(k, k)
    smooth = np.real(np.fft.ifft2(np.fft.fft2(s) * np.exp(-0.5 * (kx**2 + ky**2) * (2 * np.pi * 2.5) ** 2)))
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.3))
    panels = [(s, "atoms on sites"), (blocks, f"average over {block} × {block} boxes"),
              (smooth, "composition field c(r)")]
    for a, (img, title) in zip(ax, panels):
        im = a.imshow(img, cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                      extent=(0, crop, crop, 0))
        a.set_title(title, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    for a in ax[1:]:
        for v in range(0, crop + 1, block):
            a.axhline(v, color="white", lw=0.3); a.axvline(v, color="white", lw=0.3)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Ag fraction")
    out = ROOT / "assets" / "L28-coarse-grain.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/l28-regimes")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    data = {}
    for label, tt, c0 in CASES:
        f = cache / f"{label}.npz"
        if not f.exists():
            np.savez_compressed(f, **run(tt, c0))
        data[label] = np.load(f)
    fig, ax = plt.subplots(len(CASES), len(SNAPS), figsize=(9.0, 7.2))
    titles = {"nucleation": "nucleation\nAg 3%, T/Tc = 0.7",
              "spinodal": "spinodal\nAg 50%, T/Tc = 0.7",
              "mixed": "mixed\nAg 20%, T/Tc = 1.3"}
    for i, (label, _, _) in enumerate(CASES):
        for j, t in enumerate(SNAPS):
            a = ax[i, j]
            a.imshow(data[label][str(t)], cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(f"{t:,} sweeps\n≈ {seconds(t)}", fontsize=10)
            if j == 0:
                a.set_ylabel(titles[label], fontsize=10)
    fig.tight_layout()
    out = ROOT / "assets" / "L28-regimes.png"
    fig.savefig(out, dpi=300)
    print("wrote", out)
    coarse_grain(data["spinodal"]["10000"])
    fig, a = plt.subplots(figsize=(3.0, 3.0))
    a.imshow(data["spinodal"]["100000"], cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
    a.axis("off")
    fig.savefig(ROOT / "assets" / "L28-spinodal-snapshot.png", dpi=300, bbox_inches="tight")
    summary = {}
    for label, _, _ in CASES:
        g = data[label]["100000"]
        unlike = (g != np.roll(g, 1, 0)).sum() + (g != np.roll(g, 1, 1)).sum()
        summary[label] = round(float(unlike / (2 * g.size)), 3)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
