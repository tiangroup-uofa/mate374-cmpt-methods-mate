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
def fixed_question(mo):
    mo.md(r"""
    ## L06 · Same equation, different iterations
    **Predict:** if two rearrangements have the same root, must both iterations approach it?

    Solve $F(x)=3x^2-x+e^{2x}-2=0$ by iterating $x_{k+1}=g(x_k)$. Here $x$ is dimensionless. Start near the positive root, then try a negative start. Check the original residual even when the cobweb stops moving.
    """)
    return


@app.cell(hide_code=True)
def fixed_model(np):
    def fixed_residual(x):
        return 3*x**2 - x + np.exp(2*x) - 2

    return (fixed_residual,)


@app.cell(hide_code=False)
def student_iteration(fixed_residual):
    def student_map(x):
        # Edit this function, then select "Your map" below.
        return x - 0.1 * fixed_residual(x)

    return (student_map,)


@app.cell(hide_code=True)
def fixed_algorithm(fixed_residual, np, student_map):
    def fixed_map(x, choice, relaxation=0.1):
        if choice == "Constant c = λ":
            return x - relaxation * fixed_residual(x)
        if choice == "c = −x/2":
            return 1.5*x**3 - 0.5*x**2 + 0.5*x*np.exp(2*x)
        if choice == "c = 1/(3x)":
            return (x - np.exp(2*x) + 2) / (3*x)
        if choice == "Logarithm":
            return 0.5*np.log(-3*x**2 + x + 2)
        if choice == "Positive square root":
            return np.sqrt((x - np.exp(2*x) + 2) / 3)
        if choice == "Negative square root":
            return -np.sqrt((x - np.exp(2*x) + 2) / 3)
        if choice == "Newton–Raphson":
            return x - fixed_residual(x) / (6*x - 1 + 2*np.exp(2*x))
        return student_map(x)


    def trace_fixed(choice, x0, relaxation=0.1, maxiter=60, xtol=1e-8, ftol=1e-7):
        seq = [float(x0)]
        for k in range(maxiter):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                trial = float(fixed_map(np.float64(seq[-1]), choice, relaxation))
            if not np.isfinite(trial):
                return seq, "Stopped: map undefined or arithmetic overflow"
            seq.append(trial)
            if abs(trial) > 1e6:
                return seq, "Stopped: iterate magnitude exceeded 10⁶"
            with np.errstate(over="ignore", invalid="ignore"):
                residual = abs(fixed_residual(trial))
            small_step = abs(seq[-1] - seq[-2]) <= xtol
            if small_step and residual <= ftol:
                return seq, "Converged: step and original residual are small"
            if small_step:
                return seq, "Stagnated: small step, but original residual is too large"
        return seq, "Iteration limit: convergence not established"


    fixed_roots = np.array([-0.6046844913399143, 0.34385322840253885])
    return fixed_map, fixed_roots, trace_fixed


@app.cell(hide_code=True)
def fixed_controls(mo):
    fixed_choice = mo.ui.dropdown(
        ["Constant c = λ", "c = −x/2", "c = 1/(3x)", "Logarithm", "Positive square root", "Negative square root", "Newton–Raphson", "Your map"],
        value="Constant c = λ", label="Iteration g(x)")
    fixed_start = mo.ui.number(-1.0, 1.0, value=0.2, step=0.05, label="Initial x")
    fixed_lambda = mo.ui.slider(-1.0, 0.5, step=0.05, value=0.1, label="λ (constant-c map only)", show_value=True)
    fixed_steps = mo.ui.number(1, 30, step=1, value=8, label="Show updates")
    mo.vstack([mo.hstack([fixed_choice, fixed_start], wrap=True), fixed_lambda, fixed_steps])
    return fixed_choice, fixed_lambda, fixed_start, fixed_steps


@app.cell(hide_code=True)
def fixed_result(fixed_choice, fixed_lambda, fixed_start, trace_fixed):
    fixed_sequence, fixed_status = trace_fixed(fixed_choice.value, fixed_start.value, fixed_lambda.value)
    return fixed_sequence, fixed_status


@app.cell(hide_code=True)
def fixed_plot(
    fixed_choice,
    fixed_lambda,
    fixed_map,
    fixed_residual,
    fixed_roots,
    fixed_sequence,
    fixed_start,
    fixed_status,
    fixed_steps,
    mo,
    np,
    plt,
    trace_fixed,
):
    def draw_fixed_trace(choice, start, relaxation, steps):
        seq, status = trace_fixed(choice, start, relaxation)
        fig, ax = plt.subplots(figsize=(7, 4.1), layout="constrained")
        xx = np.linspace(-1.0, 1.0, 2001)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            yy = fixed_map(xx, choice, relaxation)
            derivative = (fixed_map(xx+1e-5, choice, relaxation) - fixed_map(xx-1e-5, choice, relaxation)) / 2e-5
        ax.plot(xx, yy, color="#007c41", label="g(x)")
        ax.plot(xx, xx, color="0.4", ls="--", label="y = x")
        # Sampled slope bands illustrate local attraction; they are not a proof
        # that g maps the entire band back into itself.
        attractive = np.isfinite(yy) & np.isfinite(derivative) & (np.abs(derivative) < 1)
        shade_label = "Sampled |g′| < 1 near a true fixed point"
        for root in fixed_roots:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                maps_root = abs(fixed_map(root, choice, relaxation) - root) < 1e-6
            idx = int(np.argmin(abs(xx-root)))
            if maps_root and attractive[idx]:
                left = right = idx
                while left > 0 and attractive[left-1]:
                    left -= 1
                while right < len(xx)-1 and attractive[right+1]:
                    right += 1
                ax.axvspan(xx[left], xx[right], alpha=0.10, color="#007c41", label=shade_label)
                shade_label = None
            if maps_root:
                ax.scatter([root], [root], color="black", s=35, zorder=5)
        shown = seq[:steps+1]
        for x, y in zip(shown[:-1], shown[1:]):
            ax.plot([x, x, y], [x, y, y], "o-", color="#d87700", lw=1.1, markersize=3)
        ax.set(xlim=(-1,1), ylim=(-1,1), xlabel="x", ylabel="g(x)", title="Vertical to g(x), horizontal to y = x")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        return fig


    def iteration_equations(choice, relaxation):
        formulas = {
            "c = −x/2": r"\frac{3}{2}x^3-\frac{1}{2}x^2+\frac{1}{2}xe^{2x}",
            "c = 1/(3x)": r"\frac{x-e^{2x}+2}{3x}",
            "Logarithm": r"\frac{1}{2}\ln(-3x^2+x+2)",
            "Positive square root": r"+\sqrt{\frac{x-e^{2x}+2}{3}}",
            "Negative square root": r"-\sqrt{\frac{x-e^{2x}+2}{3}}",
            "Newton–Raphson": r"x-\frac{3x^2-x+e^{2x}-2}{6x-1+2e^{2x}}",
        }
        notes = {
            "c = 1/(3x)": r"Domain condition: $x\ne0$.",
            "Logarithm": r"Domain condition: $-3x^2+x+2>0$.",
            "Positive square root": r"Domain condition: $(x-e^{2x}+2)/3\ge0$.",
            "Negative square root": r"Domain condition: $(x-e^{2x}+2)/3\ge0$.",
            "Newton–Raphson": r"The denominator must be nonzero.",
        }
        if choice == "Constant c = λ":
            formula = r"x-\lambda(3x^2-x+e^{2x}-2)" + rf"\quad\left(\lambda={relaxation:.2f}\right)"
            note = r"The slider supplies the current value of the constant $\lambda$."
        elif choice == "Your map":
            formula = r"\texttt{student\_map}(x)"
            note = "Edit `student_map(x)` in the visible code cell to change this update."
        else:
            formula = formulas[choice]
            note = notes.get(choice, "")
        return mo.vstack([
            mo.md("### Selected iteration: **" + choice + "**"),
            mo.hstack([
                mo.md("**Residual**\n\n$$F(x)=3x^2-x+e^{2x}-2$$"),
                mo.md("**Update**\n\n$$x_{k+1}=g(x_k)$$\n\n$$g(x)=" + formula + "$$"),
            ], widths=[1, 2], align="start", gap=2, wrap=True),
            mo.md(note),
        ])


    fixed_equations = iteration_equations(fixed_choice.value, fixed_lambda.value)
    fixed_figure = draw_fixed_trace(fixed_choice.value, fixed_start.value, fixed_lambda.value, fixed_steps.value)
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_final_residual = abs(fixed_residual(fixed_sequence[-1]))
    fixed_converged = fixed_status.startswith("Converged:")
    fixed_last_step = abs(fixed_sequence[-1] - fixed_sequence[-2]) if len(fixed_sequence) > 1 else None
    fixed_step_text = f"{fixed_last_step:.3g}" if fixed_last_step is not None else "no valid update"
    fixed_outcome = mo.callout(
        mo.md(
            f"### {'Converged' if fixed_converged else 'Not converged'}\n\n"
            f"{fixed_status.split(': ', 1)[1]}.\n\n"
            f"**Last x:** {fixed_sequence[-1]:.8g} · **|F(x)|:** {fixed_final_residual:.3g} · "
            f"**Last step:** {fixed_step_text}\n\n"
            f"Full run: {len(fixed_sequence)-1} updates. Convergence requires both "
            r"$|x_{k+1}-x_k|\le10^{-8}$ and $|F(x_{k+1})|\le10^{-7}$."
        ),
        kind="success" if fixed_converged else "warn",
    )
    mo.vstack([
        fixed_equations,
        fixed_outcome,
        fixed_figure,
        mo.md(f"The cobweb shows the first **{min(fixed_steps.value, len(fixed_sequence)-1)}** updates. "
              "Green bands show a **sampled local slope condition** near a true fixed point. A convergence guarantee also needs an interval that maps into itself. The plot stays on [−1, 1]; divergent iterates may leave it."),
        mo.accordion({"Experiments": mo.md("Compare **λ = 0.10** and **λ = −1.00** from x₀ = 0.20. Then select **c = −x/2** with x₀ = 0: does a motionless cobweb solve F(x)=0? Compare the two square-root branches. To try your own function, edit `student_map` and select **Your map**; verify its algebra and check F at the result.")}),
    ])
    return


if __name__ == "__main__":
    app.run()
