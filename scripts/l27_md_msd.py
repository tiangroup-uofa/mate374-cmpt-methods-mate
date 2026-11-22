"""L27: mean-squared displacement and temperature of the Cu-Ag MD run, from the LAMMPS log.

    uv run python scripts/l27_md_msd.py data/l27/run1000.log
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(log):
    rows, on = [], False
    for line in Path(log).read_text().splitlines():
        parts = line.split()
        if parts[:2] == ["Step", "Time"] and "c_msd[4]" in parts:
            on = True
            continue
        if on:
            try:
                rows.append([float(p) for p in parts[:7]])
            except ValueError:
                on = False
    return np.array(rows)


def main(log):
    d = read(log)
    t_ns, T, msd = d[:, 1] / 1000, d[:, 2], d[:, 6]
    hop = 2.56 ** 2          # one hop to a nearest-neighbour site, Å^2
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(t_ns, msd, color="#1f4e79", lw=1.5, label="MD: mean-squared displacement")
    ax.axhline(hop, color="#b5473a", ls="--", lw=1.2)
    ax.text(0.02, hop + 0.25, "one hop to a neighbouring site (6.6 Å²)", color="#b5473a", fontsize=9)
    ax.set(xlabel="time (ns)", ylabel="⟨R²⟩ (Å²)", ylim=(0, 8), xlim=(0, t_ns.max()))
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Cu–20 at% Ag at {T.mean():.0f} K: atoms vibrate, none hop", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "L27-md-msd.png", dpi=300)
    print(f"{len(d)} rows, MSD {msd[1:].min():.3f}-{msd.max():.3f} Å², T = {T.mean():.1f} K")


if __name__ == "__main__":
    main(sys.argv[1])
