# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget>=0.9", "traitlets>=5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import anywidget
    import traitlets
    import marimo as mo
    import struct
    import math
    from decimal import Decimal
    return Decimal, anywidget, math, mo, struct, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Flip a bit. Predict the number.

    Choose an example or type a value. Then change the sign, an exponent bit,
    or a fraction bit. The decimal readout follows the bits immediately.
    """)
    return


@app.cell(hide_code=True)
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
          controls.className = 'toolbar';
          const label = document.createElement('label');
          label.textContent = 'Format ';
          const select = document.createElement('select');
          for (const width of [32, 64]) {
            const option = document.createElement('option');
            option.value = width; option.textContent = 'float' + width;
            select.append(option);
          }
          label.append(select); controls.append(label);
          const entry = document.createElement('form');
          entry.className = 'entry';
          const inputLabel = document.createElement('label');
          inputLabel.textContent = 'Decimal value';
          const input = document.createElement('input');
          input.type = 'text'; input.spellcheck = false;
          input.autocomplete = 'off';
          inputLabel.append(input);
          const store = document.createElement('button');
          store.type = 'submit'; store.textContent = 'Store value';
          entry.append(inputLabel, store);
          const message = document.createElement('div');
          message.className = 'message'; message.setAttribute('role', 'status');
          const readout = document.createElement('output');
          readout.className = 'readout'; readout.setAttribute('aria-live', 'polite');
          const fields = document.createElement('div');
          el.append(controls, entry, message, readout, fields);
          entry.onsubmit = event => {
            event.preventDefault();
            const text = input.value.trim();
            const decimal = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;
            const special = /^[+-]?(?:inf(?:inity)?|nan)$/i;
            if (!decimal.test(text) && !special.test(text)) {
              message.textContent = 'Enter a decimal number, such as -0.3 or 1e-30 (or Infinity / NaN).';
              input.setAttribute('aria-invalid', 'true');
              return;
            }
            const value = /nan/i.test(text) ? NaN
              : /inf/i.test(text) ? (text.startsWith('-') ? -Infinity : Infinity)
              : Number(text);
            const width = model.get('state').width;
            commit(width, encode(value, width));
          };
          function commit(width, bits) {
            model.set('state', {width, bits}); model.save_changes();
            draw(); // Also reset the input when the bit pattern is unchanged.
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
          for (const [name, value] of [['0.75', 0.75], ['0.3', 0.3], ['0.1', 0.1], ['1', 1], ['0', 0]]) {
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
            const value = decode(bits);
            const display = Object.is(value, -0) ? '-0'
              : Number.isFinite(value) ? value.toPrecision(width === 32 ? 9 : 17)
              : String(value);
            input.value = display;
            input.removeAttribute('aria-invalid');
            message.textContent = '';
            readout.textContent = 'Stored float' + width + ': ' + display;
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
        .float-bits { color: #243443; font: 15px system-ui; padding: 1rem;
          border: 1px solid #d9e1e7; border-radius: 14px; background: #f8fafb; }
        .float-bits .toolbar, .float-bits .entry { display: flex; flex-wrap: wrap;
          gap: .4rem; align-items: end; margin-bottom: .8rem; }
        .float-bits label { display: flex; gap: .4rem; align-items: center; }
        .float-bits .entry label { flex: 1; min-width: 12rem; flex-direction: column;
          align-items: stretch; font-size: .85rem; }
        .float-bits button, .float-bits select, .float-bits input { font: inherit;
          padding: .45rem .6rem; border: 1px solid #b7c6d1; border-radius: 7px;
          background: #fff; color: inherit; }
        .float-bits button, .float-bits select { cursor: pointer; }
        .float-bits input { min-width: 0; font-family: monospace; }
        .float-bits .entry button { background: #28618a; color: white; }
        .float-bits .readout { display: block; padding: .8rem; background: #eaf1f6;
          border-radius: 8px; font: 600 1.05rem monospace; overflow-wrap: anywhere; }
        .float-bits .message { color: #b54d32; margin-bottom: .4rem; }
        .float-bits fieldset { margin: .8rem 0 0; border: 1px solid #d9e1e7;
          border-radius: 8px; padding: .5rem; min-width: 0; }
        .float-bits legend { font-size: .85rem; font-weight: 600; padding: 0 .3rem; }
        .float-bits fieldset button { font-family: monospace; width: 2rem;
          padding: .4rem 0; margin: .15rem; }
        .float-bits .sign { --bit-color: #aa472f; border-left: 4px solid var(--bit-color); }
        .float-bits .exponent { --bit-color: #28618a; border-left: 4px solid var(--bit-color); }
        .float-bits .fraction { --bit-color: #487333; border-left: 4px solid var(--bit-color); }
        .float-bits button[aria-pressed=true] { background: var(--bit-color);
          color: white; font-weight: bold; border-color: var(--bit-color); }
        .float-bits :focus-visible { outline: 3px solid #d48b16; outline-offset: 2px; }
        @media (prefers-color-scheme: dark) {
          .float-bits { color: #e3eaf0; background: #18232c; border-color: #465764; }
          .float-bits button, .float-bits select, .float-bits input { background: #202c36;
            border-color: #607381; }
          .float-bits .readout { background: #263b4b; }
          .float-bits fieldset { border-color: #607381; }
          .float-bits .message { color: #ffb099; }
        }
        """
    return (FloatBits,)


@app.cell(hide_code=True)
def _(FloatBits, mo):
    bit_widget = mo.ui.anywidget(FloatBits())
    bit_widget
    return (bit_widget,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Try these checks:**

    1. Reset to 1. Turn on the rightmost fraction bit. The increase is
       2⁻²³ in float32 and 2⁻⁵² in float64.
    2. Reset to 0. Turn on the rightmost fraction bit. You have made the
       smallest positive subnormal value, not a normal value with a leading 1.
    3. At 0, turn on every exponent bit. What happens if you then turn on
       a fraction bit?

    Switching format preserves the current value when possible, rounding
    if needed. It does not reinterpret the old bit string or recover lost digits.
    Choose **0.3** again after switching to compare fresh decimal inputs.
    Typed decimal input is first parsed at the browser's float64 precision,
    then rounded to float32 if selected. The readout uses 9 or 17 significant
    decimal digits; open **Exact stored value** for the full decimal expansion.
    """)
    return


if __name__ == "__main__":
    app.run()
