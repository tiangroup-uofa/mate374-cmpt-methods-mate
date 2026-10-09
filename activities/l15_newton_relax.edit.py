# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def controls(mo):
    model_choice = mo.ui.dropdown(
        options={"Harmonic spring": "harmonic", "Lennard-Jones": "lj", "Morse": "morse"},
        value="Harmonic spring", label="Bond model", full_width=True,
    )
    a_slider = mo.ui.slider(
        0.8, 1.8, value=1.0, step=0.02, label="Half-separation a",
        show_value=True, full_width=True,
    )
    x0_slider = mo.ui.slider(
        -1.5, 1.5, value=0.3, step=0.05, label="Start x₀",
        show_value=True, full_width=True,
    )
    y0_slider = mo.ui.slider(
        -1.2, 1.2, value=0.1, step=0.05, label="Start y₀",
        show_value=True, full_width=True,
    )
    return a_slider, model_choice, x0_slider, y0_slider


@app.cell(hide_code=True)
def pair_models(np):
    # Each function returns phi(r), phi'(r), and phi''(r).
    R0 = 1.2
    EPS = 0.02
    SIGMA = R0 / 2**(1/6)
    ALPHA = 5.0

    def harmonic(r, k=1.0):
        return 0.5*k*(r - R0)**2, k*(r - R0), k + 0*r

    def lennard_jones(r):
        s6 = (SIGMA/r)**6
        return (4*EPS*(s6**2 - s6),
                24*EPS*(s6 - 2*s6**2)/r,
                24*EPS*(26*s6**2 - 7*s6)/r**2)

    def morse(r):
        e = np.exp(-ALPHA*(r - R0))
        return (EPS*((1 - e)**2 - 1),
                2*EPS*ALPHA*e*(1 - e),
                2*EPS*ALPHA**2*e*(2*e - 1))

    PAIRS = {"harmonic": harmonic, "lj": lennard_jones, "morse": morse}
    return PAIRS, R0


@app.cell(hide_code=True)
def supplied_functions(PAIRS, model_choice, np):
    pair = PAIRS[model_choice.value]

    def fixed_atoms(a):
        return np.array([[-a, 0.0], [a, 0.0]])

    def energy(p, a):
        return sum(pair(np.linalg.norm(d))[0] for d in p - fixed_atoms(a))

    def gradient(p, a):
        g = np.zeros(2)
        for d in p - fixed_atoms(a):
            r = np.linalg.norm(d)
            g += pair(r)[1]*d/r
        return g

    def hessian(p, a):
        H = np.zeros((2, 2))
        for d in p - fixed_atoms(a):
            r = np.linalg.norm(d)
            _, dphi, d2phi = pair(r)
            nn = np.outer(d, d)/r**2
            H += d2phi*nn + dphi/r*(np.eye(2) - nn)
        return H

    return energy, fixed_atoms, gradient, hessian, pair


@app.cell(hide_code=True)
def newton_loop(R0, fixed_atoms, gradient, hessian, np):
    def newton(p0, a, tol=1e-10, max_steps=50):
        p = np.array(p0, dtype=float)
        path = [p.copy()]
        for m in range(max_steps + 1):
            # Safety stops: the atom must stay finite, inside the region, and outside the cores.
            if not np.all(np.isfinite(p)):
                return np.array(path), "Stopped: a Newton step is non-finite."
            if np.min(np.linalg.norm(p - fixed_atoms(a), axis=1)) < 0.5*R0:
                return np.array(path), "Stopped: the atom came too close to a fixed atom."
            if np.max(np.abs(p)) > 4.0:
                return np.array(path), "Stopped: a Newton step sent the atom out of the region."
            g = gradient(p, a)
            H = hessian(p, a)
            if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
                return np.array(path), "Stopped: non-finite derivatives."
            if np.linalg.norm(g) < tol:
                return np.array(path), "Converged: the gradient is below the tolerance."
            if m == max_steps:
                return np.array(path), "Stopped: maximum number of Newton steps reached."
            try:
                delta_p = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                return np.array(path), "Stopped: the Hessian became singular."
            p = p + delta_p
            path.append(p.copy())

    return (newton,)


@app.cell(hide_code=True)
def run_newton(a_slider, hessian, newton, np, x0_slider, y0_slider):
    a = a_slider.value
    p_start = np.array([x0_slider.value, y0_slider.value])
    path, message = newton(p_start, a)
    # Classify the stopping point by the curvature of the energy there.
    if not message.startswith("Converged:"):
        outcome = "failed"
    else:
        curvatures = np.linalg.eigvalsh(hessian(path[-1], a))
        if np.all(curvatures > 0):
            outcome = "minimum"
        elif np.all(curvatures < 0):
            outcome = "maximum"
        else:
            outcome = "saddle"
    return a, message, outcome, p_start, path


@app.cell(hide_code=True)
def step_control(mo, path):
    iteration = mo.ui.number(
        start=0, stop=len(path)-1, step=1, value=0,
        label="Inspect step m", full_width=True,
        disabled=len(path) == 1,
    )
    return (iteration,)


@app.cell(hide_code=True)
def selected_step(R0, a, energy, fixed_atoms, gradient, hessian, iteration, np, path):
    m = min(int(iteration.value), len(path)-1)
    p_current = path[m]
    safe_current = (
        np.all(np.isfinite(p_current))
        and np.min(np.linalg.norm(p_current - fixed_atoms(a), axis=1)) >= 0.5*R0
    )
    if safe_current:
        E_current = energy(p_current, a)
        g_current = gradient(p_current, a)
        H_current = hessian(p_current, a)
    else:
        E_current = g_current = H_current = None
    # Show the correction actually applied at this step, not a fabricated final step.
    delta_current = path[m+1] - p_current if m < len(path)-1 else None
    return E_current, H_current, delta_current, g_current, m, p_current, safe_current


@app.cell(hide_code=True)
def energy_map(a, np, pair):
    half_width = a + 0.9
    xs = np.linspace(-half_width, half_width, 281)
    ys = np.linspace(-1.4, 1.4, 201)
    X, Y = np.meshgrid(xs, ys)
    rL, rR = np.hypot(X + a, Y), np.hypot(X - a, Y)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        E = np.ma.masked_invalid(pair(rL)[0] + pair(rR)[0])
    # Colour by log10(E - E_min + offset). The LJ and Morse walls rise many
    # orders of magnitude above the well depth, so a linear scale would hide
    # the wells. The offset keeps the logarithm finite at the grid minimum.
    dE = E - E.min()
    offset = 1e-3*float(np.ma.median(dE))
    Z = np.ma.log10(dE + offset)
    low = float(Z.min())
    high = min(float(Z.max()), np.log10(float(np.ma.median(dE))) + 1.5)
    levels = np.linspace(low, high, 22)
    return X, Y, Z, half_width, levels


@app.cell(hide_code=True)
def plot_newton(
    R0, X, Y, Z, a, energy, fixed_atoms, half_width,
    levels, m, np, p_current, p_start, path, plt, safe_current,
):
    def draw():
        with plt.rc_context({"font.size": 11, "axes.titlesize": 11}):
            fig, (ax, cut) = plt.subplots(
                1, 2, figsize=(8.6, 3.6),
                gridspec_kw={"width_ratios": [1.45, 1]},
            )
            fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.90, wspace=0.32)
            ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="max")
            ax.contour(X, Y, Z, levels=levels[2:-1:3], colors="white",
                       linewidths=0.4, linestyles="solid")
            ax.plot(path[:, 0], path[:, 1], "--", color="#8a8f98", lw=1, zorder=3)
            ax.plot(path[:m+1, 0], path[:m+1, 1], "o-", color="#b5473a",
                    ms=3, lw=1.5, zorder=4)
            fixed_handle, = ax.plot([-a, a], [0, 0], "s", color="#231f20",
                                    ms=7, zorder=5, label="Fixed atoms")
            start_handle, = ax.plot(*p_start, "D", color="#eedfbc", mec="#231f20",
                                    ms=7, zorder=6, label="Start")
            visible = (np.all(np.isfinite(p_current))
                       and abs(p_current[0]) <= half_width
                       and abs(p_current[1]) <= 1.4)
            current_handle, = ax.plot(
                *(p_current if visible else [np.nan, np.nan]), "*",
                color="#ffcf40", mec="#231f20", mew=0.8, ms=15,
                zorder=7, label=f"Current (m = {m})",
            )
            if not visible:
                ax.text(
                    0.5, 0.96, "Current atom is off map",
                    transform=ax.transAxes, ha="center", va="top",
                    color="#b5473a", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                )
            ax.set(
                xlim=(-half_width, half_width), ylim=(-1.4, 1.4), aspect="equal",
                xlabel="x", ylabel="y", title="Energy and Newton path",
            )
            if safe_current and np.max(np.abs(p_current)) <= 4.0:
                s = np.linspace(-0.35, 0.35, 151)
                E0 = energy(p_current, a)
                values = []
                for direction, color, label in [
                    (np.array([1, 0]), "#1f77b4", "x direction"),
                    (np.array([0, 1]), "#b5473a", "y direction"),
                ]:
                    positions = p_current + s[:, None]*direction
                    safe = np.min(
                        np.linalg.norm(positions[:, None, :] - fixed_atoms(a), axis=2),
                        axis=1,
                    ) >= 0.5*R0
                    dE = np.full(s.shape, np.nan)
                    dE[safe] = [energy(p, a)-E0 for p in positions[safe]]
                    cut.plot(s, dE, color=color, lw=1.8, label=label)
                    values.extend(dE[np.isfinite(dE)])
                cut.axhline(0, color="#8a8f98", lw=0.6)
                cut.axvline(0, color="#8a8f98", lw=0.6, ls=":")
                cut.plot(0, 0, "*", color="#ffcf40", mec="#231f20", ms=12, zorder=5)
                lower, upper = min(0, min(values)), max(0, max(values))
                margin = max(0.08*(upper-lower), 1e-5)
                cut.set_ylim(lower-margin, upper+margin)
                cut.legend(fontsize=9, loc="best", frameon=False)
                cut.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            else:
                cut.text(
                    0.5, 0.5, "No local cut:\ncurrent point outside\nsafe evaluation region",
                    transform=cut.transAxes, ha="center", va="center", fontsize=10,
                )
                cut.set_yticks([])
            cut.set(
                xlabel="Displacement from current atom", ylabel="ΔE",
                title=f"Energy cuts at step {m}", xlim=(-0.35, 0.35),
            )
            fig.legend(
                handles=[fixed_handle, start_handle, current_handle],
                loc="lower center", ncol=3, frameon=False, fontsize=10,
                bbox_to_anchor=(0.53, -0.01),
            )
            return fig

    newton_figure = draw()
    plt.close(newton_figure)
    return (newton_figure,)


@app.cell(hide_code=True)
def dashboard(
    E_current, H_current, a_slider, delta_current, g_current,
    iteration, m, message, mo, model_choice, newton_figure, np,
    outcome, p_current, path, x0_slider, y0_slider,
):
    def matrix_tex(values):
        if values is None:
            return r"\text{—}"
        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        def number_tex(v):
            if v != 0 and (abs(v) < 0.001 or abs(v) >= 10000):
                mantissa, exponent = f"{v:.2e}".split("e")
                return mantissa + r"\times10^{" + str(int(exponent)) + "}"
            return f"{v:.3g}"

        return (
            r"\begin{bmatrix}"
            + r"\\".join(" & ".join(number_tex(v) for v in row) for row in array)
            + r"\end{bmatrix}"
        )

    def display():
        controls_panel = mo.vstack([
            model_choice, a_slider, x0_slider, y0_slider,
            iteration,
            mo.md(f"Steps: **0–{len(path)-1}**  \nColour: log scale of E − E_min."),
        ], gap=0.65)
        norm_text = "—" if g_current is None else f"{np.linalg.norm(g_current):.2e}"
        energy_text = "—" if E_current is None else f"{E_current:.5g}"
        selected_info = mo.md(
            f"**Step {m}** · position ({p_current[0]:+.4g}, {p_current[1]:+.4g})"
            f" · E = {energy_text} · ‖∇E‖ = {norm_text}"
        )
        matrices = mo.hstack([
            mo.md(r"$$\mathbf H_m=" + matrix_tex(H_current) + "$$"),
            mo.md(r"$$\nabla E_m=" + matrix_tex(g_current) + "$$"),
            mo.md(r"$$\Delta\mathbf x_m=" + matrix_tex(delta_current) + "$$"),
        ], justify="space-around", align="center", gap=0.5)
        end_note = (
            "No next correction: the run ended at this step."
            if delta_current is None else r"$\mathbf H_m\Delta\mathbf x_m=-\nabla E_m$"
        )
        plot_panel = mo.vstack([
            mo.as_html(newton_figure), selected_info, matrices, mo.md(end_note),
        ], gap=0.1)
        steps = len(path) - 1
        x_end, y_end = path[-1]
        where = f"({x_end:+.3f}, {y_end:+.3f}) after {steps} steps"
        status, kind, background, border = {
            "minimum": (
                f"Converged to a true minimum at {where}. "
                "The energy rises in every direction.",
                "success", "#eef8ef", "#2e7d32",
            ),
            "saddle": (
                f"Converged to a saddle point at {where}. "
                "The forces vanish, but the energy falls along some direction.",
                "warn", "#f6ece2", "#8b5a2b",
            ),
            "maximum": (
                f"Converged to an energy maximum at {where}. "
                "The forces vanish, but the energy falls in every direction.",
                "warn", "#f6ece2", "#8b5a2b",
            ),
            "failed": (
                f"Not converged. {message} ({steps} steps.)",
                "danger", "#fdecea", "#b5473a",
            ),
        }[outcome]
        return mo.vstack([
            mo.callout(status, kind=kind).style({
                "background-color": background,
                "border-color": border,
                "border-radius": "4px",
            }),
            mo.hstack(
                [controls_panel, plot_panel], widths=[1, 3.5],
                align="start", gap=1.2,
            ),
        ], gap=0.5)

    display()
    return


if __name__ == "__main__":
    app.run()
