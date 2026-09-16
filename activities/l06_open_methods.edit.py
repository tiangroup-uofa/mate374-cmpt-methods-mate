# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def course_imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq, minimize_scalar

    return mo, np, plt


@app.cell(hide_code=True)
def open_question(mo):
    mo.md(r"""
    ## L06 · Where will the next point land?
    **Predict:** with no bracket to keep the search in place, can changing the starting volumes send us to a different root?

    Use the residual $F(V)=P(V)-50\ \mathrm{bar}$, where $P(V)$ is the vdW pressure. The physical domain is $V>0.04267$ L.
    """)
    return


@app.cell(hide_code=True)
def open_algorithm(np):
    def open_residual(V):
        return 0.08314462618 * 280 / (V - 0.04267) - 3.592 / V**2 - 50


    def open_derivative(V):
        return -0.08314462618 * 280 / (V - 0.04267)**2 + 2 * 3.592 / V**3


    def trace_open(method, x0, x1, maxiter=30, xtol=1e-8, ftol=1e-6):
        """Trace secant/Newton–Raphson steps; report unsafe steps without clipping."""
        points = [float(x0)] if method == "Newton–Raphson" else [float(x0), float(x1)]
        lines = []
        if any(x <= 0.04267 for x in points):
            return points, lines, "Outside the physical domain"
        for k in range(maxiter):
            x = points[-1]
            fx = open_residual(x)
            if fx == 0:
                return points, lines, "Exact zero"
            if method == "Newton–Raphson":
                slope = open_derivative(x)
            else:
                old = points[-2]
                if x == old:
                    return points, lines, "Repeated starting points: secant slope undefined"
                slope = (fx - open_residual(old)) / (x - old)
            if not np.isfinite(slope) or abs(slope) < 1e-12:
                return points, lines, "Slope too small: step unreliable"
            trial = x - fx / slope
            lines.append((x, fx, slope, trial))
            if not np.isfinite(trial) or trial <= 0.04267:
                return points, lines, f"Stopped: proposed V={trial:.5g} L leaves the domain"
            points.append(float(trial))
            if abs(trial - x) <= xtol and abs(open_residual(trial)) <= ftol:
                return points, lines, "Converged: step and residual tolerances met"
        return points, lines, "Iteration limit: convergence not established"

    return open_residual, trace_open


@app.cell(hide_code=True)
def open_controls(mo):
    open_method = mo.ui.dropdown(["Secant", "Newton–Raphson"], value="Secant", label="Method")
    open_x0 = mo.ui.number(0.045, 1.0, value=0.20, step=0.005, label="First volume (L)")
    open_x1 = mo.ui.number(0.045, 1.0, value=0.21, step=0.005, label="Second volume, secant only (L)")
    open_step = mo.ui.number(1, 15, step=1, value=1, label="Show update")
    mo.vstack([open_method, open_x0, open_x1, open_step])
    return open_method, open_step, open_x0, open_x1


@app.cell(hide_code=True)
def open_result(open_method, open_step, open_x0, open_x1, trace_open):
    open_points, open_lines, open_status = trace_open(open_method.value, open_x0.value, open_x1.value)
    open_visible = min(open_step.value, len(open_lines))
    return open_lines, open_points, open_status, open_visible


@app.cell(hide_code=True)
def open_plot(
    mo,
    np,
    open_lines,
    open_method,
    open_points,
    open_residual,
    open_status,
    open_visible,
    plt,
):
    def draw_open_trace(points, lines, visible, method):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
        volumes = np.linspace(0.055, 0.65, 1400)
        axes[0].plot(volumes, open_residual(volumes), color="#007c41", label="Residual F(V)")
        axes[0].axhline(0, color="0.4", lw=1)
        axes[0].set(
            xlim=(0.055, 0.65), ylim=(-25, 30), xlabel="Volume (L)",
            ylabel="Residual (bar)", title="Current points → new point",
        )

        shown = points[:visible + (1 if method == "Newton–Raphson" else 2)]
        if visible and lines:
            x, fx, slope, trial = lines[visible - 1]
            if method == "Secant":
                current = points[visible - 1:visible + 1]
                point_label = "two newest points"
            else:
                current = points[visible - 1:visible]
                point_label = "current point"
            current_values = [open_residual(value) for value in current]
            axes[0].scatter(current, current_values, s=42, color="#303f9f", zorder=4, label=point_label)

            finite_trial = np.isfinite(trial)
            line_points = current + ([trial] if finite_trial else [])
            line_left = max(0.055, min(line_points))
            line_right = min(0.65, max(line_points))
            if line_left < line_right:
                line_x = np.linspace(line_left, line_right, 100)
                intercept = fx - slope * x
                axes[0].plot(
                    line_x, slope * line_x + intercept,
                    color="#d87700", alpha=0.8, lw=1.5, label="line to proposed point",
                )
            if finite_trial and trial > 0.04267:
                axes[0].scatter(
                    [trial], [open_residual(trial)], s=58, color="#d87700",
                    zorder=5, label="new point",
                )
                if 0.055 <= trial <= 0.65:
                    axes[0].axvline(trial, color="#d87700", ls=":", lw=1)
                axes[0].annotate(
                    "new point", (trial, open_residual(trial)),
                    xytext=(4, 6), textcoords="offset points", fontsize=9,
                )
            axes[0].legend(fontsize=8, loc="upper right")

        axes[1].semilogy(
            range(len(shown)),
            np.maximum(np.abs([open_residual(v) for v in shown]), 1e-14),
            "o-", color="#303f9f",
        )
        axes[1].axhline(1e-6, ls="--", color="0.5", label="Residual tolerance")
        axes[1].set(
            xlabel="Evaluated point index", ylabel="|F(V)| (bar)",
            title="Is the pressure mismatch shrinking?",
        )
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.2)
        return fig


    open_figure = draw_open_trace(open_points, open_lines, open_visible, open_method.value)
    open_current = open_points[min(open_visible + (0 if open_method.value == "Newton–Raphson" else 1), len(open_points)-1)]
    mo.vstack([
        open_figure,
        mo.md(f"**Shown:** {open_visible} update(s), V = **{open_current:.8f} L**, F = **{open_residual(open_current):+.3g} bar**.\n\n"
              f"**Full run:** {open_status}. Last accepted V = **{open_points[-1]:.8f} L**; |F| = **{abs(open_residual(open_points[-1])):.3g} bar**.\n\n"
              "The plot window stays fixed; an open step can leave the window or the physical domain. No step is clipped back into place."),
        mo.accordion({"Try these starts": mo.md("Secant: **(0.08, 0.085)**, **(0.11, 0.12)**, **(0.20, 0.21)** L. Which root does each find? Then try **(0.15, 0.16)** L. For Newton–Raphson, compare **0.20** and **0.60** L. Bring the secant starting points close together: does only the first step resemble Newton–Raphson, or the whole history?")}),
    ])
    return


if __name__ == "__main__":
    app.run()
