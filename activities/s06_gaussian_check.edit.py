# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## S06 · Check your hand elimination

    Complete the row operations and back substitution on paper. Select the matching
    question, then replace the three entries of `x_trial` with your answer. The
    residual checks your answer against the **original equations**, before any swaps.

    **Predict:** if you swap two coefficient rows but forget to swap their right-hand
    sides, will a correctly executed back substitution still pass this check?
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    question = mo.ui.dropdown(["Question 1", "Question 2"], value="Question 1",
                              label="Hand calculation")
    question
    return (question,)


@app.cell
def problem_data(np, question):
    if question.value == "Question 1":
        A = np.array([[2., 1., -1.], [4., 5., 0.], [-2., 2., 7.]])
        b = np.array([5., 14., -5.])
    else:
        A = np.array([[0., 2., 1.], [2., 1., -1.], [4., 4., 1.]])
        b = np.array([0., -1., 2.])
    return A, b


@app.cell
def your_answer(np):
    # Replace these values with the result of your back substitution.
    x_trial = np.array([0.0, 0.0, 0.0])
    return (x_trial,)


@app.cell
def residual_check(A, b, mo, np, x_trial):
    residual = A @ x_trial - b
    relative_residual = np.linalg.norm(residual)/np.linalg.norm(b)
    mo.vstack([
        mo.ui.table([
            {"Equation": i + 1, "Left side A @ x": float((A @ x_trial)[i]),
             "Right side b": float(b[i]), "Residual": float(residual[i])}
            for i in range(3)
        ], selection=None),
        mo.md(f"""
        **Relative residual:** {relative_residual:.3e}.

        All three residuals should be zero, apart from rounding. If they are not,
        compare your first transformed row with the original augmented matrix,
        including the right-hand side.
        """),
    ])
    return relative_residual, residual


if __name__ == "__main__":
    app.run()
