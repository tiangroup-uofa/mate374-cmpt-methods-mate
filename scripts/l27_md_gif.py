"""L27 MD animation: one (100) layer of the Cu-20 at% Ag LAMMPS run at 1000 K.

Reads a LAMMPS custom dump (id type x y z), keeps the atoms of one (100) layer, and
writes assets/L27-cuag-md.gif plus a 300 dpi PNG of the last frame. Ag is dark,
Cu is light.

    uv run python scripts/l27_md_gif.py path/to/dump.cuag
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DT_PS = 0.002     # time step, ps
N_LAYERS = 40     # (100) layers in 20 fcc cells


def read_dump(path):
    frames = []
    with open(path) as f:
        while f.readline():
            step = int(f.readline()); f.readline()
            n = int(f.readline()); f.readline()
            box = np.array([list(map(float, f.readline().split()[:2])) for _ in range(3)])
            f.readline()
            atoms = np.loadtxt([f.readline() for _ in range(n)])
            frames.append((step, box, atoms))
    return frames


def draw(step, box, atoms, dpi):
    L = box[:, 1] - box[:, 0]
    z = (atoms[:, 4] - box[2, 0]) / L[2]
    layer = atoms[np.abs(z - 0.5) < 0.6 / N_LAYERS]
    fig, ax = plt.subplots(figsize=(4.2, 4.5), dpi=dpi)
    colours = np.where(layer[:, 1] == 2, "#1f4e79", "#d9a066")
    ax.scatter(layer[:, 2], layer[:, 3], s=16, c=colours, linewidths=0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"t = {step * DT_PS / 1000:.2f} ns    Cu–20 at% Ag, 1000 K", fontsize=10)
    fig.tight_layout()
    return fig


def main(dump):
    frames = read_dump(dump)
    images = []
    for step, box, atoms in frames:
        fig = draw(step, box, atoms, dpi=90)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    gif = ROOT / "assets" / "L27-cuag-md.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=200, loop=0, optimize=True)
    step, box, atoms = frames[-1]
    fig = draw(step, box, atoms, dpi=300)
    fig.savefig(ROOT / "assets" / "L27-cuag-md-final.png", dpi=300)
    print(len(frames), "frames, last at", step * DT_PS / 1000, "ns")


if __name__ == "__main__":
    main(sys.argv[1])
