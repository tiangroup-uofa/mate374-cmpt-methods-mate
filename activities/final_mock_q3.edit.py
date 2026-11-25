# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock final · Q3")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # Q3 · Cooling a steel plate

    Use the helper below if you want to find the time at which the plate
    reaches 350 K by root finding. Choose the number of Euler steps `n`.
    Each call starts the plate at 1200 K. Add your work in a new cell.
    """)
    return


@app.cell
def _():
    def final_temp(t_final, n):
        """Return the plate temperature (K) after n Euler steps to t_final (s)."""
        h = t_final / n
        temperature = 1200.0
        for _ in range(n):
            temperature += h * (-0.2 * temperature + 60.0)
        return temperature

    return (final_temp,)


if __name__ == "__main__":
    app.run()
