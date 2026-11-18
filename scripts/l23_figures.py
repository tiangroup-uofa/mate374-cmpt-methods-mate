"""Exact spring motion for the L23 cosine case study.

Run with the course Python environment. All animation frames and the static
fallback are rendered at 300 dpi. The other L23 schematics have editable SVG
sources in assets/ and do not need regeneration by this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def main():
    red = "#b5473a"
    gray = "#5e5e5e"
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"], "font.size": 12}):
        fig, (spring_ax, curve_ax) = plt.subplots(1, 2, figsize=(7.1, 2.25), layout="constrained")
        spring_ax.set(xlim=(-2.7, 1.5), ylim=(-0.8, 0.8))
        spring_ax.axis("off")
        spring_ax.add_patch(plt.Rectangle((-2.55, -.42), .16, .84, facecolor="0.9", edgecolor=gray, hatch="////"))
        spring_ax.axvline(0, ymin=.2, ymax=.8, color="0.65", lw=1, ls="--")
        spring_ax.text(0, -.58, "Equilibrium", ha="center", color=gray)
        spring_ax.annotate("", xy=(1.3, -.36), xytext=(-1.4, -.36),
                           arrowprops={"arrowstyle": "->", "color": gray})
        spring_ax.text(1.35, -.4, "y", style="italic")
        spring_line, = spring_ax.plot([], [], color=gray, lw=1.5)
        atom, = spring_ax.plot([], [], "o", color=red, markersize=18)
        spring_ax.text(-2.55, .62, "a", weight="bold", fontsize=16)
        curve_ax.text(-.15, 1.03, "b", weight="bold", fontsize=16, transform=curve_ax.transAxes)
        t = np.linspace(0, 2 * np.pi, 400)
        curve_ax.plot(t, np.cos(t), color="0.7", lw=1.2)
        curve_ax.set(xlim=(-.15, 2*np.pi+.25), ylim=(-1.3, 1.35), xlabel="Time t", ylabel="Displacement y")
        curve_ax.set_xticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
        curve_ax.set_yticks([-1, 0, 1])
        curve_ax.spines[["top", "right"]].set_visible(False)
        curve_ax.axhline(0, color="0.85", lw=.7)
        trace, = curve_ax.plot([], [], color=red, lw=2)
        marker, = curve_ax.plot([], [], "o", color=red, markersize=6)

        frames = []
        for theta in np.linspace(0, 2 * np.pi, 80, endpoint=False):
            q = np.cos(theta)
            x = np.r_[-2.39, -2.19, np.linspace(-2.1, q-.3, 17), q-.18, q-.12]
            y = np.r_[0., 0., .13 * np.where(np.arange(17) % 2, 1., -1.), 0., 0.]
            spring_line.set_data(x, y)
            atom.set_data([q], [0])
            progress = np.linspace(0, theta, max(2, round(theta*50)))
            trace.set_data(progress, np.cos(progress))
            marker.set_data([theta], [q])
            fig.canvas.draw()
            image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            # Canvas pixel density is set to 300 dpi before any frame is captured.
            frames.append(image)
        frames[0].save(ASSETS / "L23-spring-motion.png", dpi=(300, 300))
        frames[0].save(ASSETS / "L23-spring-motion.gif", save_all=True,
                       append_images=frames[1:], duration=60, loop=0, disposal=2)
        plt.close(fig)


if __name__ == "__main__":
    # Set the actual canvas resolution used by buffer_rgba(), including all GIF frames.
    plt.rcParams["figure.dpi"] = 300
    main()
