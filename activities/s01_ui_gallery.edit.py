# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # A small marimo UI gallery

    These controls are ready to use in a notebook. Open **Edit code** to
    see their names and constructors. You do not need to memorize the
    options or build these controls from scratch in course activities.
    """)
    return


@app.cell
def _(mo):
    def _build_ui_gallery():
        slider = mo.ui.slider(
            start=0,
            stop=100,
            value=50,
            label="Slider",
            show_value=True,
        )
        number = mo.ui.number(
            start=0,
            stop=2000,
            value=933,
            label="Number input",
        )
        dropdown = mo.ui.dropdown(
            options=["aluminum", "copper", "steel"],
            value="aluminum",
            label="Dropdown",
        )
        text = mo.ui.text(
            value="Al-4.5Cu",
            label="Text input",
        )
        switch = mo.ui.switch(
            value=True,
            label="Switch",
        )
        checkbox = mo.ui.checkbox(
            value=False,
            label="Checkbox",
        )
        radio = mo.ui.radio(
            options=["solid", "liquid", "two phase"],
            value="solid",
            label="Radio buttons",
            inline=True,
        )
        multiselect = mo.ui.multiselect(
            options=["temperature", "energy", "composition"],
            value=["temperature"],
            label="Multiple selection",
        )

        return mo.vstack(
            [
                mo.hstack([slider, number], widths="equal", gap=2),
                mo.hstack([dropdown, text], widths="equal", gap=2),
                mo.hstack([switch, checkbox], justify="start", gap=3),
                radio,
                multiselect,
            ],
            gap=1.5,
        )

    _build_ui_gallery()
    return


if __name__ == "__main__":
    app.run()
