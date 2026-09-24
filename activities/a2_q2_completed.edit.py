# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.11"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A2 Q2 · Completed reference notebook

    Let $x$ be the atomic fraction of A in an A–B alloy, with $0<x<1$.
    The dimensionless parameter $H$ measures the energetic penalty for mixing
    relative to the thermal contribution. The equilibrium compositions satisfy

    $$
    F(x;H)=\ln\frac{x}{1-x}+H(1-2x)=0.
    $$
    The map and optional iteration loop are completed. Change H to compare
    the intersections, or use the one-update calculator at H = 2.5.
    """)
    return


@app.cell(hide_code=True)
def plotting_support(mo, np, plt):
    def plot_map(mapping, H):
        try:
            xs = np.linspace(0.001, 0.999, 1001)
            values = [mapping(float(x)) for x in xs]
            if any(value is Ellipsis for value in values):
                return mo.md("Replace `...` in `g(x)` in Q2.2 to display the plot.")
            ys = np.asarray(values, dtype=float)
            if ys.shape != xs.shape or not np.all(np.isfinite(ys)):
                raise ValueError("Return one finite number for each input x.")
            center = float(mapping(0.5))
            if not np.isfinite(center):
                raise ValueError("g(0.5) must be finite.")
        except Exception as error:
            return mo.callout(mo.md(f"Check `g(x)`: {error}"), kind="warn")
        fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
        ax.plot(xs, ys, color="#007c41", label=r"$y=g(x;H)$")
        ax.plot(xs, xs, "--", color="0.35", label=r"$y=x$")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Composition x", ylabel="y",
               title=f"Your fixed-point map at H = {H:g}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.close(fig)
        if not np.isclose(center, 0.5, rtol=0, atol=1e-8):
            return mo.vstack([
                mo.callout(mo.md(
                    f"**Check your expression:** g(0.5) = {center:.8g}. "
                    "The curves do not intersect at x = 0.5."
                ), kind="danger"),
                fig,
            ])
        return fig

    return (plot_map,)


@app.cell(hide_code=True)
def model_parameter(strength):
    H = float(strength.value)
    return (H,)


@app.cell(hide_code=True)
def model_residual(np):
    def F(x, H):
        return np.log(x / (1.0 - x)) + H * (1.0 - 2.0 * x)

    return (F,)


@app.cell(hide_code=True)
def map_instructions(mo):
    mo.md(r"""
    ## Q2.2 · Enter your fixed-point form

    Rearranging $F(x;H)=0$ gives $x=g(x;H)$.
    **`g(x)` in the cell below** implements the rearrangement. The global parameter `H`
    is set by the Q2.3 slider and starts at 2.5. NumPy is available as `np`.
    Run the cell to draw $y=g(x;H)$ and $y=x$ immediately below it.
    **Do the curves intersect at $x=0.5$?** This checks the central root from Q2.1.
    """)
    return


@app.cell(hide_code=False)
def student_map(H, np):
    def g(x):
        return 1 / (1 + np.exp(H * (1 - 2*x)))

    return (g,)


@app.cell(hide_code=True)
def central_plot(H, g, plot_map):
    plot_map(g, H)
    return


@app.cell(hide_code=True)
def comparison_instructions(mo):
    mo.md(r"""
    ## Q2.3 · Compare intersections

    Move the slider to **$H=1.5$**, then **$H=2.5$**. For each value,
    do you see an intersection in $0.5<x<1$? Record your observations.
    The plot uses your expression from Q2.2.
    """)
    return


@app.cell(hide_code=True)
def comparison_control(mo):
    strength = mo.ui.slider(
        0.5, 3.0, step=0.1, value=2.5,
        label="Mixing parameter H", show_value=True,
    )
    strength
    return (strength,)


@app.cell(hide_code=True)
def comparison_plot(H, g, plot_map):
    plot_map(g, H)
    return


@app.cell(hide_code=True)
def iteration_instructions(mo):
    mo.md(r"""
    ## Q2.4 · One fixed-point update

    Set the Q2.3 slider to **$H=2.5$** and start with $x_0=0.9$.
    Enter the current $x_n$ to obtain **one** update using your `g(x)`, as in Seminar 3.
    Record the next value and copy it into the input to continue. Keep enough
    digits when copying to avoid rounding each step to the stopping tolerance.

    Count the updates and stop when $|x_{n+1}-x_n|<10^{-4}$. Report the final
    composition to five significant digits and its residual. The calculator
    displays $F(x_n)$ and $F(x_{n+1})$ so you can check both values.
    """)
    return


@app.cell(hide_code=True)
def composition_input(mo):
    current_x = mo.ui.number(
        value=0.9, step=1e-10, label="Current composition xₙ (0 < xₙ < 1)",
    )
    current_x
    return (current_x,)


@app.cell(hide_code=True)
def single_update(F, H, current_x, g, mo, np):
    mo.stop(not np.isclose(H, 2.5), mo.md("Set the Q2.3 slider to H = 2.5 for this calculation."))
    try:
        _x = float(current_x.value)
        if not np.isfinite(_x) or not 0 < _x < 1:
            raise ValueError("Enter a finite composition strictly between 0 and 1.")
        _next = g(_x)
        if _next is Ellipsis:
            calculation = mo.md("Complete `g(x)` in Q2.2 to use the calculator.")
        else:
            _next = float(_next)
            if not np.isfinite(_next) or not 0 < _next < 1:
                raise ValueError("Your map must return a finite composition strictly between 0 and 1.")
            calculation = mo.md(
                f"**Next composition xₙ₊₁:** `{_next:.12g}`\n\n"
                "| Quantity | Value |\n|---|---:|\n"
                f"| Step: ∣xₙ₊₁ − xₙ∣ | {abs(_next - _x):.8g} |\n"
                f"| F(xₙ; H=2.5) | {F(_x, 2.5):+.8g} |\n"
                f"| F(xₙ₊₁; H=2.5) | {F(_next, 2.5):+.8g} |"
            )
    except Exception as error:
        calculation = mo.callout(mo.md(f"**No valid update.** {error}"), kind="warn")
    calculation
    return


@app.cell(hide_code=True)
def optional_instructions(mo):
    mo.md(r"""
    ## Optional · Write your own iteration loop

    If you want to automate your updates, complete `fixed_point` below.
    Return the full sequence $[x_0,x_1,\ldots,x_n]$, including the starting value.
    Use the step tolerance from Q2.4 and limit the number of updates with `maxiter`.
    [Lecture 6's minimal fixed-point loop](https://tiangroup-uofa.github.io/mate374-cmpt-methods-mate/units/02/L06/#generalizing-to-fixed-point-iteration)
    provides a starting point. Adapt its stopping test to Q2.4.

    **This coding exercise is optional.** You can complete Q2.4 using the
    one-update calculator above. If you write a loop, include your code in your submission.
    """)
    return


@app.cell(hide_code=False)
def optional_loop():
    def fixed_point(g, x0, tol=1e-4, maxiter=100):
        sequence = [x0]
        for _ in range(maxiter):
            next_x = g(sequence[-1])
            step = abs(next_x - sequence[-1])
            sequence.append(next_x)
            if step < tol:
                return sequence
        raise RuntimeError("Step tolerance not reached")

    return (fixed_point,)


@app.cell(hide_code=True)
def completed_sequence(F, H, fixed_point, g, mo, np):
    mo.stop(not np.isclose(H, 2.5), mo.md("Set H = 2.5 to display the Q2.4 reference sequence."))
    sequence = fixed_point(g, 0.9)
    mo.vstack([
        mo.md(f"## Q2.4 · Reference result\n\n{len(sequence)-1} updates; "
              f"xβ = {sequence[-1]:.5g}; F(xβ) = {F(sequence[-1], H):+.8g}."),
        mo.ui.table([{"n": n, "x_n": x, "F(x_n)": F(x, H)}
                     for n, x in enumerate(sequence)], selection=None),
    ])
    return (sequence,)


if __name__ == "__main__":
    app.run()
