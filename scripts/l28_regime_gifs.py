"""Animated versions of the three L28 regimes (log-spaced frames from 10 to 100,000 sweeps).

    uv run python scripts/l28_regime_gifs.py --cache /tmp/l28-regime-gifs
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l28_regimes as R

FRAMES = tuple(sorted(set(int(round(10 ** (1 + 4 * k / 39))) for k in range(40))))


def render(label, snaps, out):
    images = []
    title = {"nucleation": "nucleation · 3% Ag · T/Tc = 0.7",
             "spinodal": "spinodal · 50% Ag · T/Tc = 0.7",
             "mixed": "mixed · 20% Ag · T/Tc = 1.3"}[label]
    for t in FRAMES:
        fig, ax = plt.subplots(figsize=(3.6, 3.9), dpi=90)
        ax.imshow(snaps[str(t)], cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n{t:,} sweeps ≈ {R.seconds(t)}", fontsize=9)
        fig.tight_layout(); fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    images[0].save(out, save_all=True, append_images=images[1:], duration=[150] * (len(images) - 1) + [1500],
                   loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/l28-regime-gifs")
    ap.add_argument("--case")
    args = ap.parse_args()
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    if args.case:
        case = [c for c in R.CASES if c[0] == args.case][0]
        np.savez_compressed(cache / f"{args.case}.npz", **R.run(case[1], case[2], times=FRAMES))
        return
    procs = [subprocess.Popen([sys.executable, __file__, "--cache", str(cache), "--case", c[0]])
             for c in R.CASES if not (cache / f"{c[0]}.npz").exists()]
    for p in procs:
        p.wait()
    for label, _, _ in R.CASES:
        render(label, np.load(cache / f"{label}.npz"), R.ROOT / "assets" / f"L28-{label}.gif")
        print("wrote", label)


if __name__ == "__main__":
    main()
