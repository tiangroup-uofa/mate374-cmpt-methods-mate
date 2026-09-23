# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget>=0.9", "traitlets>=5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import anywidget
    import traitlets
    import marimo as mo
    import struct
    import math
    from decimal import Decimal
    return Decimal, anywidget, math, mo, struct, traitlets


@app.cell
def _(mo):
    mo.md("""
    # A1: compare floating-point representations

    Use **0.75** and **0.1** in float32 and float64 for Question 2.3.
    Inspect the significand and exponent in each format. Expand the exact
    stored value to check whether the decimal input is represented exactly.

    After changing format, click the number's button again to encode a
    fresh input rather than carry over an already-rounded value.
    """)
    return


@app.cell
def _(anywidget, traitlets):
    class FloatBits(anywidget.AnyWidget):
        # One synchronized state keeps the format and its bits together.
        state = traitlets.Dict(default_value={
            "width": 32, "bits": "00111111010000000000000000000000"
        }).tag(sync=True)
        _esm = r"""
        function render({model, el}) {
          el.classList.add('float-bits');
          const controls = document.createElement('div');
          const label = document.createElement('label');
          label.textContent = 'Format ';
          const select = document.createElement('select');
          for (const width of [32, 64]) {
            const option = document.createElement('option');
            option.value = width; option.textContent = 'float' + width;
            select.append(option);
          }
          label.append(select); controls.append(label);
          const fields = document.createElement('div');
          el.append(controls, fields);
          function commit(width, bits) {
            model.set('state', {width, bits}); model.save_changes();
          }
          function encode(value, width) {
            const view = new DataView(new ArrayBuffer(width / 8));
            if (width === 32) view.setFloat32(0, value, false);
            else view.setFloat64(0, value, false);
            return Array.from(new Uint8Array(view.buffer),
              byte => byte.toString(2).padStart(8, '0')).join('');
          }
          function decode(bits) {
            const bytes = new Uint8Array(bits.length / 8);
            for (let i = 0; i < bytes.length; i++)
              bytes[i] = parseInt(bits.slice(i * 8, i * 8 + 8), 2);
            const view = new DataView(bytes.buffer);
            return bits.length === 32 ? view.getFloat32(0, false) : view.getFloat64(0, false);
          }
          select.onchange = () => {
            const state = model.get('state');
            const width = Number(select.value);
            commit(width, encode(decode(state.bits), width));
          };
          for (const [name, value] of [['0.75', 0.75], ['0.1', 0.1], ['1', 1], ['0', 0]]) {
            const button = document.createElement('button');
            button.textContent = name;
            button.onclick = () => {
              const width = model.get('state').width;
              commit(width, encode(value, width));
            };
            controls.append(button);
          }
          function draw() {
            const {width, bits} = model.get('state');
            select.value = width;
            // Keep keyboard focus on the flipped bit after rebuilding.
            const focused = fields.contains(document.activeElement)
              ? document.activeElement.dataset.index : null;
            fields.replaceChildren();
            const e = width === 32 ? 8 : 11;
            for (const [name, start, end, cls] of [
              ['Sign', 0, 1, 'sign'], ['Exponent', 1, 1 + e, 'exponent'],
              ['Fraction (leading 1 is implicit for normal values)', 1 + e, width, 'fraction']]) {
              const group = document.createElement('fieldset');
              group.className = cls;
              const legend = document.createElement('legend');
              legend.textContent = name; group.append(legend);
              for (let i = start; i < end; i++) {
                const button = document.createElement('button');
                button.textContent = bits[i]; button.dataset.index = i;
                button.setAttribute('aria-pressed', bits[i] === '1');
                const bitLabel = cls === 'fraction' ? 'Fraction weight 2^-' + (i - e)
                  : cls === 'exponent' ? 'Exponent weight 2^' + (end - 1 - i) : 'Sign bit';
                button.setAttribute('aria-label', bitLabel);
                button.title = bitLabel;
                button.onclick = () => {
                  const current = model.get('state');
                  const next = current.bits.slice(0, i) + (current.bits[i] === '0' ? '1' : '0')
                    + current.bits.slice(i + 1);
                  commit(current.width, next);
                };
                group.append(button);
              }
              fields.append(group);
            }
            if (focused !== null) fields.querySelector(`[data-index="${focused}"]`)?.focus();
          }
          model.on('change:state', draw); draw();
          return () => model.off('change:state', draw);
        }
        export default {render};
        """
        _css = """
        .float-bits { color: #243443; font: 15px system-ui; }
        .float-bits button, .float-bits select { font: inherit; padding: .35rem .55rem;
          margin: .15rem; border: 1px solid #8796a3; border-radius: 4px;
          background: #fff; color: inherit; cursor: pointer; }
        .float-bits fieldset { margin: .7rem 0; border: 1px solid #8796a3;
          border-radius: 6px; padding: .4rem; }
        .float-bits fieldset button { font-family: monospace; min-width: 2rem; }
        .float-bits .sign { border-left: 5px solid #b54d32; }
        .float-bits .exponent { border-left: 5px solid #277897; }
        .float-bits .fraction { border-left: 5px solid #65833d; }
        .float-bits button[aria-pressed=true] { background: #dcebf4; font-weight: bold; }
        .float-bits button:focus-visible { outline: 3px solid #d48b16; outline-offset: 1px; }
        @media (prefers-color-scheme: dark) {
          .float-bits { color: #e3eaf0; }
          .float-bits button, .float-bits select { background: #202c36; }
          .float-bits button[aria-pressed=true] { background: #385568; }
        }
        """
    return (FloatBits,)


@app.cell
def _(FloatBits, mo):
    bit_widget = mo.ui.anywidget(FloatBits())
    bit_widget
    return (bit_widget,)


@app.cell
def _(Decimal, bit_widget, math, mo, struct):
    def explain_bits(state):
        width, bits = state["width"], state["bits"]
        exponent_count, fraction_count, bias = (8, 23, 127) if width == 32 else (11, 52, 1023)
        sign = int(bits[0])
        exponent = int(bits[1:1 + exponent_count], 2)
        fraction = int(bits[1 + exponent_count:], 2)
        value = struct.unpack(">f" if width == 32 else ">d",
                              int(bits, 2).to_bytes(width // 8, "big"))[0]
        if exponent == 2**exponent_count - 1:
            kind = "NaN" if fraction else "infinity"
            formula = "All exponent bits are 1: zero fraction means infinity; otherwise NaN."
        elif exponent == 0:
            kind = "signed zero" if fraction == 0 else "subnormal"
            formula = f"(-1)^{sign} × ({fraction}/2^{fraction_count}) × 2^{1 - bias}"
        else:
            kind = "normal"
            formula = f"(-1)^{sign} × (1 + {fraction}/2^{fraction_count}) × 2^({exponent} - {bias})"
        header = mo.md(f"""
        **{kind}** · decimal readout: `{value!r}`

        Stored exponent **E = {exponent}**; fraction integer **F = {fraction}**.

        `{formula}`
        """)
        if math.isfinite(value):
            numerator, denominator = value.as_integer_ratio()
            details = mo.accordion({"Exact stored value (not a rounded display)": mo.md(
                f"**Ratio:** `{numerator} / {denominator}`\n\n"
                f"**Decimal:** `{Decimal.from_float(value)}`"
            )})
        else:
            details = mo.md("Infinity and NaN do not have finite integer ratios.")
        return mo.vstack([header, details])

    explain_bits(bit_widget.value["state"])
    return


@app.cell
def _(mo):
    mo.md("""
    For these normal numbers, the significand is **1.fraction bits** and
    the exponent is **p = E − bias**. The bias is 127 for float32 and 1023
    for float64. Including the implicit leading 1, the formats provide
    24 and 53 significant binary digits, respectively.

    Switching format preserves the current value when possible, rounding
    if needed. It does not recover lost digits. Click **0.1** again after
    switching to compare fresh decimal inputs.
    """)
    return


if __name__ == "__main__":
    app.run()
