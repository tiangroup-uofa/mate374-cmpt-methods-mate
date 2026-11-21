"""L25 time-step comparison: shared physical time and L21's 3D renderer.

Run: uv run python scripts/l25_timestep_movie.py
Reuses L24's actual initial-state and velocity-Verlet functions.
"""
import ast
import json
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]


def notebook_functions():
    tree = ast.parse((ROOT / "activities/l24_cluster_velocity_verlet.edit.py").read_text())
    ns = {
        "np": np, "optimize": optimize,
        "params": {"atoms": 13, "added": 0.3,
                   "preparation": "Random velocities at relaxed structure"},
    }
    # Execute the selected cell bodies, not the notebook UI or export machinery.
    for marker in ("def energy_func", "pos_guess =", "def initial_state",
                   "def velocity_verlet_step"):
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and marker in ast.unparse(node)]
        assert len(matches) == 1, marker
        body = [node for node in matches[0].body if not isinstance(node, ast.Return)]
        exec(compile(ast.Module(body=body, type_ignores=[]), "<L24 cell>", "exec"), ns)
    return ns


def main():
    l21 = runpy.run_path(str(ROOT / "scripts/l21_figures.py"))
    ns = notebook_functions()
    tau_fs = 3.405e-10 * np.sqrt(
        39.948 * 1.66054e-27 / (119.8 * 1.380649e-23)
    ) * 1e15
    steps_fs = (1, 20, 200)
    # Every frame is an actual saved state for ALL three step sizes.
    frame_times_fs = np.arange(0, 8001, 200, dtype=int)
    trajectories = {}
    checks = {}
    for dt_fs in steps_fs:
        n_steps = int(frame_times_fs[-1] // dt_fs)
        positions, velocities, stopped = ns["run"](
            ns["pos0"], ns["vel0"], dt_fs / tau_fs, n_steps
        )
        assert not stopped, "Select a shared horizon before the safety stop."
        assert len(positions) == n_steps + 1
        energy = np.array([ns["U"](p) for p in positions])
        energy += 0.5 * np.sum(velocities**2, axis=(1, 2))
        error = 100 * np.abs(energy - energy[0]) / abs(energy[0])
        trajectories[dt_fs] = (positions, error)
        assert np.all(frame_times_fs % dt_fs == 0)
        checks[dt_fs] = {
            "steps_at_8_ps": n_steps,
            "max_relative_energy_error_percent": float(error.max()),
            "atoms_outside_fixed_view_at_8_ps":
                int(np.count_nonzero(np.any(np.abs(positions[-1]) > 4, axis=1))),
        }
    assert checks[1]["steps_at_8_ps"] == 20 * checks[20]["steps_at_8_ps"]
    assert checks[20]["max_relative_energy_error_percent"] < 0.1
    assert checks[200]["max_relative_energy_error_percent"] > 100
    print(json.dumps(checks, indent=2))

    with plt.rc_context(l21["STYLE"]):
        fig = plt.figure(figsize=(10, 4.4))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        fig.subplots_adjust(left=0, right=1, top=0.83, bottom=0.20, wspace=0)
        clock = fig.suptitle("", fontsize=16, y=0.98)
        # Fixed footer positions prevent labels moving when atoms leave the camera.
        counts = [fig.text((i + .5) / 3, .16, "", ha="center", fontsize=13)
                  for i in range(3)]
        energy_labels = [fig.text((i + .5) / 3, .10, "", ha="center", fontsize=11)
                         for i in range(3)]
        statuses = [fig.text((i + .5) / 3, .035, "", ha="center", fontsize=12,
                            weight="bold") for i in range(3)]

        def draw(frame):
            time_fs = int(frame_times_fs[frame])
            clock.set_text(f"Same physical time: t = {time_fs / 1000:.1f} ps")
            for i, (ax, dt_fs) in enumerate(zip(axes, steps_fs)):
                index = time_fs // dt_fs
                positions, errors = trajectories[dt_fs]
                ax.cla()
                outside = l21["draw_cluster"](ax, positions[index], size=230, view=4.0)
                ax.set_title(f"Δt = {dt_fs} fs", fontsize=15, pad=0)
                counts[i].set_text(f"{index:,} integration steps")
                err = errors[index]
                energy_labels[i].set_text(
                    "|ΔE| / |E₀| < 0.001%" if 0 < err < 0.001
                    else (f"|ΔE| / |E₀| = {err:.3f}%" if err < 100
                          else "|ΔE| / |E₀| > 100%")
                )
                if i == 0:
                    statuses[i].set_text("20× more steps than 20 fs")
                    statuses[i].set_color(l21["INK"])
                elif i == 1:
                    statuses[i].set_text("Cluster intact")
                    statuses[i].set_color(l21["GREEN"])
                else:
                    statuses[i].set_text(
                        f"Breakup: {outside}/13 outside view" if outside
                        else ("Large energy error" if err > 10 else "Initially intact")
                    )
                    statuses[i].set_color(
                        l21["RED"] if outside or err > 10 else l21["GREY"]
                    )
            return []

        draw(len(frame_times_fs) - 1)
        fig.savefig(ROOT / "assets/L25-cluster-steps-poster.png",
                    dpi=300, facecolor="white")
        # Repeat identical frames only for start/end pauses; all panels remain synchronized.
        frames = [0] * 4 + list(range(len(frame_times_fs))) + [len(frame_times_fs) - 1] * 10
        animation = FuncAnimation(fig, draw, frames=frames, interval=125)
        animation.save(ROOT / "assets/L25-cluster-steps.gif",
                       writer=PillowWriter(fps=8), dpi=300)
        plt.close(fig)
    return checks


if __name__ == "__main__":
    main()
