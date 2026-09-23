# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # A1 playground: cutoff error versus retained work

    This reference example uses a **one-dimensional chain of 12 argon
    atoms**, not the three-dimensional cluster in A1. Every unique pair
    contributes to the reference energy. The cutoff calculation keeps
    only pairs separated by $r\leq r_c$.

    Move the cutoff and observe what happens to the retained pair count
    and the error. Open **Edit code** to inspect how unique pairs,
    distances, energies, and the cutoff mask are represented.
    """)
    return


@app.cell
def _(mo):
    cutoff_A = mo.ui.slider(
        start=4.0,
        stop=30.0,
        step=1.0,
        value=8.0,
        label="Cutoff radius, r_c (Å)",
        show_value=True,
    )
    cutoff_A
    return (cutoff_A,)


@app.cell
def _(cutoff_A, mo, np, plt):
    def _make_cutoff_study(selected_cutoff_A):
        epsilon_eV = 0.0103
        sigma_A = 3.40
        spacing_A = 4.0

        coordinates_A = np.column_stack(
            [np.arange(12, dtype=float) * spacing_A, np.zeros(12), np.zeros(12)]
        )
        pair_i, pair_j = np.triu_indices(len(coordinates_A), k=1)
        displacement_A = coordinates_A[pair_i] - coordinates_A[pair_j]
        distance_A = np.linalg.norm(displacement_A, axis=1)
        pair_energy_eV = 4.0 * epsilon_eV * (
            (sigma_A / distance_A) ** 12 - (sigma_A / distance_A) ** 6
        )
        reference_eV = np.sum(pair_energy_eV)

        selected = distance_A <= selected_cutoff_A
        truncated_eV = np.sum(pair_energy_eV[selected])
        absolute_error_eV = abs(truncated_eV - reference_eV)

        cutoff_grid_A = np.arange(4.0, 31.0, 1.0)
        retained_counts = np.array(
            [np.count_nonzero(distance_A <= cutoff) for cutoff in cutoff_grid_A]
        )
        truncated_energies_eV = np.array(
            [np.sum(pair_energy_eV[distance_A <= cutoff]) for cutoff in cutoff_grid_A]
        )
        errors_eV = np.abs(truncated_energies_eV - reference_eV)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(retained_counts, errors_eV, marker="o", linewidth=1.5)
        ax.scatter(
            np.count_nonzero(selected),
            max(absolute_error_eV, 1e-16),
            color="#d95f02",
            s=70,
            zorder=3,
            label=f"selected $r_c={selected_cutoff_A:.0f}$ Å",
        )
        ax.set_xlabel("Retained pair interactions")
        ax.set_ylabel("Absolute energy error (eV)")
        ax.set_title("Small-chain cutoff study")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        summary = mo.callout(
            mo.md(
                f"""
                ## Selected cutoff

                - all unique pairs: **{len(distance_A)}**
                - retained pairs: **{np.count_nonzero(selected)}**
                - all-pairs reference energy: **{reference_eV:.6f} eV**
                - truncated energy: **{truncated_eV:.6f} eV**
                - absolute error: **{absolute_error_eV:.3e} eV**

                A larger cutoff retains more pair-energy evaluations. Does the
                error decrease by the same factor each time another shell of
                neighbours is included?
                """
            ),
            kind="info",
        )
        return mo.vstack([summary, fig], gap=1.5)

    _make_cutoff_study(cutoff_A.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Transfer to A1

    A1 replaces this chain with a $5\times5\times5$ finite cluster. The
    ideas are the same, but you must calculate and report the assigned
    system yourself. Remember that applying a cutoff after calculating
    all distances does not make a naïve distance search cheaper.
    """)
    return


if __name__ == "__main__":
    app.run()
