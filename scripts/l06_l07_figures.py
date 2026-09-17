"""Regenerate L06/L07 static figures from the saved molab notebook snapshots.

Run: uv run python scripts/l06_l07_figures.py
No Quarto render or WASM build is required.
"""
from io import BytesIO
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def notebook_definitions(filename):
    spec = spec_from_file_location(filename.replace(".", "_"), ROOT / "activities" / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    return definitions


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(ASSETS / name)


def simple_root_function(x):
    return x**3 - 2*x**2 + 1


def bisection_sequence(left=0.0, right=1.4, steps=6):
    sequence = [left, right]
    for _ in range(steps):
        midpoint = (left + right) / 2
        sequence.append(midpoint)
        if simple_root_function(left) * simple_root_function(midpoint) <= 0:
            right = midpoint
        else:
            left = midpoint
    return sequence


def secant_sequence(first=0.0, second=1.4, steps=5):
    sequence = [first, second]
    for _ in range(steps):
        f_first = simple_root_function(first)
        f_second = simple_root_function(second)
        next_point = second - f_second * (second - first) / (f_second - f_first)
        sequence.append(next_point)
        first, second = second, next_point
    return sequence


def regula_falsi_steps(left=0.0, right=1.4, steps=2):
    updates = []
    for _ in range(steps):
        f_left, f_right = simple_root_function(left), simple_root_function(right)
        trial = (left * f_right - right * f_left) / (f_right - f_left)
        updates.append((left, right, trial))
        if f_left * simple_root_function(trial) < 0:
            right = trial
        else:
            left = trial
    return updates


def plot_simple_sequence(ax, sequence, title, bracket=None):
    x = np.linspace(-0.02, 1.45, 700)
    ax.plot(x, simple_root_function(x), color="#007c41", label="$f(x)$")
    ax.axhline(0, color="0.35", lw=1)
    ax.scatter([1], [0], marker="*", s=75, color="#303f9f", zorder=4, label="root $x=1$")
    if bracket is not None:
        ax.axvspan(*bracket, color="#007c41", alpha=0.10, label="retained bracket")
    colours = plt.cm.plasma(np.linspace(0.15, 0.9, len(sequence)))
    for index, (point, colour) in enumerate(zip(sequence, colours)):
        value = simple_root_function(point)
        ax.plot([point, point], [0, value], color=colour, ls=":", lw=0.9)
        ax.scatter([point], [value], color=colour, s=34, zorder=5)
        ax.annotate(f"$x_{{{index}}}$", (point, value), xytext=(3, 5), textcoords="offset points", fontsize=10)
    ax.set(xlim=(-0.02, 1.45), ylim=(-0.3, 1.15), xlabel="$x$", ylabel="$f(x)$", title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")


def regula_falsi_sequence(left=0.0, right=1.4, steps=7):
    states = []
    for _ in range(steps):
        f_left, f_right = simple_root_function(left), simple_root_function(right)
        trial = (left * f_right - right * f_left) / (f_right - f_left)
        if f_left * simple_root_function(trial) < 0:
            next_left, next_right = left, trial
        else:
            next_left, next_right = trial, right
        states.append((left, right, trial, next_left, next_right))
        left, right = next_left, next_right
    return states


def secant_update_states(first=0.0, second=0.5, steps=7):
    states = []
    for _ in range(steps):
        f_first, f_second = simple_root_function(first), simple_root_function(second)
        trial = second - f_second * (second - first) / (f_second - f_first)
        states.append((first, second, trial))
        first, second = second, trial
    return states


def draw_root_method_frame(ax, title):
    x = np.linspace(-0.02, 1.45, 700)
    ax.clear()
    ax.plot(x, simple_root_function(x), color="#007c41", label="$f(x)$")
    ax.axhline(0, color="0.35", lw=1)
    ax.scatter([1], [0], marker="*", s=75, color="#303f9f", zorder=4, label="root $x=1$")
    ax.set(xlim=(-0.02, 1.45), ylim=(-0.3, 1.15), xlabel="$x$", ylabel="$f(x)$", title=title)
    ax.grid(alpha=0.2)


def draw_regula_falsi_secant_frame(axes, frame, regula_states, secant_states):
    left, right, trial, next_left, next_right = regula_states[frame]
    draw_root_method_frame(axes[0], f"Closed domain (regula falsi) · step {frame + 1}")
    axes[0].axvspan(next_left, next_right, color="#007c41", alpha=0.12, label="retained bracket")
    axes[0].plot([left, right], [simple_root_function(left), simple_root_function(right)],
                 "--", color="#d87700", lw=1.5, label="line through bracket")
    axes[0].scatter([left, right], [simple_root_function(left), simple_root_function(right)],
                    color="#303f9f", s=35, zorder=5)
    axes[0].scatter([trial], [simple_root_function(trial)], color="#d87700", s=48, zorder=6,
                    label="next point")
    axes[0].axvline(trial, color="#d87700", ls=":", lw=1)
    axes[0].annotate("next point", (trial, simple_root_function(trial)),
                     xytext=(4, 6), textcoords="offset points", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right")

    first, second, trial = secant_states[frame]
    draw_root_method_frame(axes[1], f"Open domain (secant) · step {frame + 1}")
    line_left = min(first, second, trial)
    line_right = max(first, second, trial)
    f_first, f_second = simple_root_function(first), simple_root_function(second)
    slope = (f_second - f_first) / (second - first)
    intercept = f_first - slope * first
    line_x = np.linspace(line_left, line_right, 100)
    axes[1].plot(line_x, slope * line_x + intercept, "--", color="#d87700", lw=1.5,
                 label="line through newest pair")
    axes[1].scatter([first, second], [f_first, f_second], color="#303f9f", s=35, zorder=5,
                    label="newest pair")
    axes[1].scatter([trial], [simple_root_function(trial)], color="#d87700", s=48, zorder=6,
                    label="next point")
    axes[1].axvline(trial, color="#d87700", ls=":", lw=1)
    axes[1].annotate("next point", (trial, simple_root_function(trial)),
                     xytext=(4, 6), textcoords="offset points", fontsize=9)
    axes[1].legend(fontsize=8, loc="upper right")


def save_regula_falsi_secant_animation():
    regula_states = regula_falsi_sequence()
    secant_states = secant_update_states()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    frames = []
    for frame in range(len(regula_states)):
        draw_regula_falsi_secant_frame(axes, frame, regula_states, secant_states)
        fig.suptitle("Two methods for $f(x)=x^3-2x^2+1$")
        image_bytes = BytesIO()
        fig.savefig(image_bytes, format="png", dpi=300, bbox_inches="tight")
        image_bytes.seek(0)
        frames.append(Image.open(image_bytes).convert("P", palette=Image.Palette.ADAPTIVE))
        image_bytes.close()
    gif_path = ASSETS / "L06-bracketing-vs-open.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1200,
                   loop=0, optimize=True)
    print(gif_path)
    fallback_frame = min(1, len(regula_states) - 1)
    draw_regula_falsi_secant_frame(axes, fallback_frame, regula_states, secant_states)
    fig.suptitle("Two methods for $f(x)=x^3-2x^2+1$")
    fig.savefig(ASSETS / "L06-bracketing-vs-open.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for frame in frames:
        frame.close()
    print(ASSETS / "L06-bracketing-vs-open.png")


def plot_simple_line(ax, first, second, label, colour):
    f_first, f_second = simple_root_function(first), simple_root_function(second)
    slope = (f_second - f_first) / (second - first)
    intercept = f_first - slope * first
    line_x = np.linspace(min(first, second), max(first, second), 80)
    ax.plot(line_x, slope * line_x + intercept, ls="--", color=colour, lw=1.4, label=label)
    trial = second - f_second / slope
    ax.scatter([trial], [0], marker="x", s=55, color=colour, zorder=6)


def plot_two_step_comparison():
    function_x = np.linspace(-0.02, 1.45, 700)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    false_position = regula_falsi_steps()
    for ax, title in zip(axes, ["Regula falsi: retain the bracket", "Secant: retain the newest pair"]):
        ax.plot(function_x, simple_root_function(function_x), color="#007c41", label="$f(x)$")
        ax.axhline(0, color="0.35", lw=1)
        ax.scatter([1], [0], marker="*", s=75, color="#303f9f", zorder=4, label="root $x=1$")
        ax.set(xlim=(-0.02, 1.45), ylim=(-0.3, 1.15), xlabel="$x$", ylabel="$f(x)$", title=title)
        ax.grid(alpha=0.2)
    left = axes[0]
    for index, (a, b, trial) in enumerate(false_position, start=1):
        plot_simple_line(left, a, b, f"step {index} line", "#d87700" if index == 1 else "#9b59b6")
        left.scatter([trial], [simple_root_function(trial)], color="#d87700", s=35, zorder=5)
        left.annotate(f"$x_{{{index+1}}}$", (trial, simple_root_function(trial)), xytext=(3, 5), textcoords="offset points", fontsize=8)
    left.axvspan(0, false_position[-1][2], color="#007c41", alpha=0.10, label="retained after step 2")
    left.legend(fontsize=8, loc="upper right")

    right = axes[1]
    secant = secant_sequence(steps=2)
    for index, (first, second) in enumerate(zip(secant[:2], secant[1:3]), start=1):
        plot_simple_line(right, first, second, f"step {index} line", "#d87700" if index == 1 else "#9b59b6")
    for index, point in enumerate(secant):
        right.scatter([point], [simple_root_function(point)], color="#d87700", s=35, zorder=5)
        right.annotate(f"$x_{{{index}}}$", (point, simple_root_function(point)), xytext=(3, 5), textcoords="offset points", fontsize=8)
    right.legend(fontsize=8, loc="upper right")
    fig.suptitle("Two updates for $f(x)=x^3-2x^2+1$ from $x_0=0$, $x_1=1.4$")
    return fig


def save_fixed_point_figure():
    fixed = notebook_definitions("l06_fixed_point.edit.py")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1), layout="constrained")
    for ax, relaxation, label in zip(axes, [0.1, -1.0], ["Approaches the root", "Moves away from the root"]):
        fixed["draw_fixed_trace"]("Constant c = λ", 0.2, relaxation, 10, ax=ax)
        ax.set_title(f"λ = {relaxation:g}: {label}")
    save(fig, "L06-fixed-point.png")


def save_vdw_energy_figure(energy):
    """Keep the EOS connection and show pressure selecting the deeper well."""
    temperature = 280.0
    saturation = energy["co2_saturation"](temperature)
    vv = np.linspace(0.06, 0.37, 1200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    axes[0].plot(vv, energy["co2_pressure"](vv, temperature), color="0.25")
    axes[0].set(ylabel="Pressure (bar)", ylim=(35, 80), title="van der Waals EOS")

    axes[1].axhline(0, color="0.6", lw=0.8)
    for pressure, colour, style, label in [
        (50.0, "#1f77b4", "--", "50 bar: gas favoured"),
        (saturation, "#ff7f0e", "-", f"{saturation:.3f} bar: equal minima"),
        (57.5, "#d62728", "-.", "57.5 bar: liquid favoured"),
    ]:
        volumes = energy["co2_volumes"](temperature, pressure)
        pressure_label = f"$P_{{sat}}$ = {pressure:.3f} bar" if pressure == saturation else f"P = {pressure:g} bar"
        axes[0].axhline(pressure, color=colour, ls=style, label=pressure_label)
        axes[0].scatter(volumes, np.full(len(volumes), pressure), color=colour, s=25, zorder=3)
        relative_energy = energy["co2_relative_energy"](vv, temperature, pressure)
        values = energy["co2_relative_energy"](volumes, temperature, pressure)
        axes[1].plot(vv, relative_energy, color=colour, ls=style, label=label)
        axes[1].scatter(volumes, values, color=colour, s=25, zorder=3)
        if pressure == saturation:
            for volume, value, name, offset in zip(
                volumes, values,
                ["Liquid-like\nminimum", "Maximum", "Gas-like\nminimum"],
                [(4, -36), (12, 5), (0, -36)],
            ):
                axes[1].annotate(name, (volume, value), xytext=offset,
                                 textcoords="offset points", fontsize=8, ha="center")
    axes[1].set(ylabel=r"$\mathcal{G}(v;T,P)-G_{\mathrm{gas}}(T,P)$ (J/mol)",
                ylim=(-65, 190), title="Gibbs free energy and minimum")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set(xlabel="Molar volume (L/mol)", xlim=(0.06, 0.37))
        ax.grid(alpha=0.2)
    fig.suptitle("vdW CO₂ at 280 K")
    save(fig, "L07-vdw-energy.png")


def save_golden_section_figure():
    R = (np.sqrt(5) - 1) / 2
    f = lambda x: (x - 0.68)**2 + 0.3
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    a, b = 0.0, 1.0
    reused = None
    titles = ["Step 1: compare two values", "Step 2: reuse old x₂", "Step 3: reuse old x₁"]
    for ax, title in zip(axes, titles):
        x1, x2 = b - R * (b - a), a + R * (b - a)
        xx = np.linspace(0, 1, 500)
        ax.plot(xx, f(xx), color="#007c41")
        ax.axvspan(a, b, alpha=0.12, color="#007c41")
        for x in (x1, x2):
            is_reused = reused is not None and np.isclose(x, reused)
            ax.scatter(x, f(x), color="#1f77b4" if is_reused else "#ff7f0e",
                       s=40, zorder=3, label="Reused value" if is_reused else None)
        for x, label in [(a, "a"), (x1, "x₁"), (x2, "x₂"), (b, "b")]:
            ax.axvline(x, color="0.5", lw=0.7, ls=":")
            ax.text(x, 0.27, label, ha="center")
        comparison = "$f(x_1)>f(x_2)$" if f(x1) > f(x2) else "$f(x_1)<f(x_2)$"
        ax.text(0.04, 0.94, comparison, transform=ax.transAxes, va="top")
        ax.set(xlabel="$x$", ylabel="$f(x)$", title=title, ylim=(0.24, 0.82))
        if reused is not None:
            ax.legend(fontsize=8, loc="upper right")
        if f(x1) > f(x2):
            a, reused = x1, x2
        else:
            b, reused = x2, x1i
    fig.suptitle("Golden section search")
    save(fig, "L07-golden-section.png")


def main():
    roots = notebook_definitions("l06_open_methods.edit.py")
    save_regula_falsi_secant_animation()
    points, lines, _ = roots["trace_open"]("Secant", 0.20, 0.21)
    save(roots["draw_open_trace"](points, lines, 5, "Secant"), "L06-open-methods.png")

    save_fixed_point_figure()

    energy = notebook_definitions("l07_co2_free_energy.edit.py")
    save_vdw_energy_figure(energy)
    save(energy["energy_figure"], "L07-co2-demo.png")

    save_golden_section_figure()
    plt.close("all")


if __name__ == "__main__":
    main()
