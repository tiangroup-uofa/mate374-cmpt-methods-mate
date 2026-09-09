# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget", "traitlets", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import anywidget
    import traitlets
    import numpy as np
    import struct
    import re
    from fractions import Fraction

    return Fraction, anywidget, mo, np, re, struct, traitlets


@app.cell(hide_code=True)
def _(Fraction, np, re, struct):
    def conversion_steps(text, base):
        """Manual conversion uses exact rationals, never a float remainder."""
        text = text.strip()
        if len(text) > 100:
            raise ValueError("Please use at most 100 characters.")
        if base == "decimal":
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?", text):
                raise ValueError("Enter a finite decimal, such as -13.625 or 0.10.")
            exact = Fraction(text)
        else:
            if not re.fullmatch(r"[+-]?(?:[01]+(?:\.[01]*)?|\.[01]+)", text):
                raise ValueError("Enter binary digits with an optional sign and point, e.g. 1101.101.")
            unsigned = text.lstrip("+-")
            whole, _, tail = unsigned.partition(".")
            exact = Fraction(int((whole or "0") + tail, 2), 2 ** len(tail))
            if text.startswith("-"):
                exact = -exact
        if abs(exact) > 10**30 or (exact and abs(exact) < Fraction(1, 10**30)):
            raise ValueError("For readable board work, use zero or a magnitude from 1e-30 to 1e30.")

        def decimal_string(value):
            # All displayed rationals here have terminating decimal expansions.
            sign = "-" if value < 0 else ""
            numerator, denominator = abs(value).as_integer_ratio()
            integer, remainder = divmod(numerator, denominator)
            digits = []
            while remainder:
                digit, remainder = divmod(remainder * 10, denominator)
                digits.append(str(digit))
            return sign + str(integer) + ("." + "".join(digits) if digits else "")

        magnitude = abs(exact)
        integer = magnitude.numerator // magnitude.denominator
        current = integer
        divisions = []
        while current:
            quotient, remainder = divmod(current, 2)
            divisions.append([str(current), str(quotient), str(remainder)])
            current = quotient
        integer_bits = bin(integer)[2:]
        remainder = magnitude - integer
        multiplications, digits, seen = [], [], {}
        repeat_start = None
        for index in range(32):
            if not remainder:
                break
            if remainder in seen and repeat_start is None:
                repeat_start = seen[remainder]
                # Keep at least eight digits for A1 Q2.2.
                if index >= 8:
                    break
            elif remainder not in seen:
                seen[remainder] = index
            product = 2 * remainder
            bit = product.numerator // product.denominator
            new_remainder = product - bit
            multiplications.append([str(index + 1), decimal_string(remainder),
                                    decimal_string(product), str(bit), decimal_string(new_remainder)])
            digits.append(str(bit))
            remainder = new_remainder
            if repeat_start is not None and len(digits) >= 8:
                break
        fraction_bits = "".join(digits)
        sign = "-" if exact < 0 else ""
        binary = sign + integer_bits + ("." + fraction_bits if fraction_bits else "")
        if remainder:
            binary += "…"
        if repeat_start is not None:
            first_remainder = magnitude - integer
            cycle_seen, cycle_digits = {}, []
            while first_remainder not in cycle_seen:
                cycle_seen[first_remainder] = len(cycle_digits)
                doubled = 2 * first_remainder
                bit = doubled.numerator // doubled.denominator
                cycle_digits.append(str(bit))
                first_remainder = doubled - bit
            start = cycle_seen[first_remainder]
            status = f"Repeating: {''.join(cycle_digits[:start]) or '(no prefix)'} then [{''.join(cycle_digits[start:])}] repeats. Square brackets mark the repeating block."
        elif remainder:
            status = "Stopped after 32 fractional digits; a nonzero remainder remains."
        else:
            status = "Terminating: the remainder reached zero. No trailing ellipsis is needed."

        # Reverse conversion: preserve the supplied bits for binary input.
        reverse_bits = text.lstrip("+-") if base == "binary" else integer_bits + ("." + fraction_bits if fraction_bits else "")
        left, _, right = reverse_bits.partition(".")
        places = []
        total = Fraction(0)
        for bit, power in zip(left + right, range(len(left) - 1, -len(right) - 1, -1)):
            contribution = int(bit) * Fraction(2) ** power
            total += contribution
            places.append([bit, str(power), decimal_string(contribution)])
        total = -total if exact < 0 else total
        first_eight = (fraction_bits + "00000000")[:8]
        truncated = Fraction(integer) + Fraction(int(first_eight, 2), 256)
        if exact < 0:
            truncated = -truncated

        ieee = []
        for name, dtype, fmt, exp_len, bias in [
            ("float32", np.float32, ">f", 8, 127),
            ("float64", np.float64, ">d", 11, 1023),
        ]:
            stored = float(dtype(decimal_string(exact)))
            # Preserve an explicitly entered negative zero.
            if not exact and text.startswith("-"):
                stored = -0.0
            bits = "".join(f"{byte:08b}" for byte in struct.pack(fmt, stored))
            exponent = int(bits[1:1 + exp_len], 2)
            fraction = bits[1 + exp_len:]
            stored_ratio = Fraction(*stored.as_integer_ratio())
            normal = exponent != 0
            power = exponent - bias if normal else 1 - bias
            ieee.append({"name": name, "sign": bits[0], "exponent": bits[1:1 + exp_len],
                         "fraction": fraction, "biased": exponent, "bias": bias,
                         "power": power, "significand": ("1." if normal else "0.") + fraction,
                         "kind": "normal" if normal else ("zero" if stored == 0 else "subnormal"),
                         "stored": decimal_string(stored_ratio), "exact": stored_ratio == exact})
        return {"decimal": decimal_string(exact), "binary": binary, "status": status,
                "divisions": divisions or [["0", "0", "0"]], "multiplications": multiplications,
                "places": places, "reverse": decimal_string(total),
                "reverse_error": decimal_string(abs(total - exact)),
                "eight_bits": sign + integer_bits + "." + first_eight,
                "eight_decimal": decimal_string(truncated),
                "eight_error": decimal_string(abs(truncated - exact)), "ieee": ieee}

    return (conversion_steps,)


@app.cell(hide_code=True)
def _(anywidget, conversion_steps, traitlets):
    class ConversionBoard(anywidget.AnyWidget):
        text = traitlets.Unicode("2026").tag(sync=True)
        base = traitlets.Unicode("decimal").tag(sync=True)
        result = traitlets.Dict().tag(sync=True)
        _esm = r"""
        function render({model, el}) {
          el.classList.add('conversion-board');
          el.innerHTML = `
            <p>Work on the board first, then reveal a check. Changing the input hides all checks.</p>
            <form class="controls">
              <label>Input base <select name="base"><option value="decimal">Decimal</option><option value="binary">Binary</option></select></label>
              <label>Number <input name="number" type="text" maxlength="100" spellcheck="false"></label>
              <button type="submit">Prepare checks</button>
            </form>
            <div class="examples" aria-label="Example inputs"></div>
            <p class="notice" role="status" aria-live="polite"></p>
            <div class="checks"></div>`;
          const form = el.querySelector('form');
          const input = form.elements.number, base = form.elements.base;
          input.value = model.get('text'); base.value = model.get('base');
          let opened = new Set();
          const checks = el.querySelector('.checks');
          function submit(text, radix) {
            opened.clear(); checks.replaceChildren();
            el.querySelector('.notice').textContent = 'Preparing checks…';
            input.value = text; base.value = radix;
            const unchanged = text === model.get('text') && radix === model.get('base');
            model.set('text', text); model.set('base', radix); model.save_changes();
            if (unchanged) draw();
          }
          form.addEventListener('submit', event => {event.preventDefault(); submit(input.value, base.value);});
          for (const [text, radix] of [['2026','decimal'],['1100111','binary'],['0.75','decimal'],['0.10','decimal'],['-13.625','decimal']]) {
            const button = document.createElement('button'); button.type = 'button';
            button.textContent = text + (radix === 'binary' ? ' (binary)' : '');
            button.addEventListener('click', () => submit(text, radix));
            el.querySelector('.examples').append(button);
          }
          function paragraph(parent, text) {
            const p = document.createElement('p'); p.textContent = text; parent.append(p);
          }
          function table(parent, headers, rows) {
            const wrap = document.createElement('div'); wrap.className = 'table-wrap';
            const table = document.createElement('table');
            const head = table.createTHead().insertRow();
            headers.forEach(text => {const th = document.createElement('th'); th.scope = 'col'; th.textContent = text; head.append(th);});
            const body = table.createTBody();
            rows.forEach(row => {const tr = body.insertRow(); row.forEach(text => {tr.insertCell().textContent = text;});});
            wrap.append(table); parent.append(wrap);
          }
          function section(title, fill) {
            const details = document.createElement('details');
            details.open = opened.has(title);
            const summary = document.createElement('summary'); summary.textContent = title;
            details.append(summary); fill(details); checks.append(details);
            details.addEventListener('toggle', () => {if(details.open) opened.add(title); else opened.delete(title);});
          }
          function draw() {
            const data = model.get('result'); checks.replaceChildren();
            el.querySelector('.notice').textContent = data.error || 'Checks ready. Open only the step you have worked through.';
            if (data.error || !data.divisions) return;
            section('1 · Integer part: divide by 2', panel => {
              paragraph(panel, 'Convert the magnitude first. Read the remainders from bottom to top; attach the minus sign afterward. This is signed notation, not a fixed-width two’s-complement encoding.');
              table(panel, ['Value','Quotient','Remainder'], data.divisions);
            });
            section('2 · Fractional part: multiply by 2', panel => {
              paragraph(panel, 'Read the integer bits from top to bottom. These steps use the exact input, not a rounded machine float.');
              if(data.multiplications.length) table(panel, ['Step','Fraction','× 2','Bit','New fraction'], data.multiplications);
              else paragraph(panel, 'No fractional part.');
              paragraph(panel, data.status);
              paragraph(panel, 'Combined binary: ' + data.binary);
            });
            section('3 · Back to decimal: sum the place values', panel => {
              paragraph(panel, 'Each contribution is bit × 2^power. Sum the contributions, then restore the sign. For a repeating decimal input, this checks only the displayed finite binary prefix.');
              table(panel, ['Bit','Power','Contribution'], data.places);
              paragraph(panel, 'Sum with sign = ' + data.reverse + '; exact input = ' + data.decimal + '; absolute error = ' + data.reverse_error);
            });
            section('4 · Q2.2: keep eight fractional bits', panel => {
              paragraph(panel, 'Discard digits beyond the eighth place (truncate, not round). Pad a shorter terminating fraction with zeros.');
              paragraph(panel, data.eight_bits + ' (binary) = ' + data.eight_decimal + ' (decimal)');
              paragraph(panel, 'Absolute error against the exact input: ' + data.eight_error);
            });
            section('5 · Q2.3: normalized form and IEEE fields', panel => {
              paragraph(panel, 'Finite IEEE values are rounded to the nearest representable number, with ties to even. This is different from discarding digits in Q2.2.');
              for(const value of data.ieee) {
                const heading = document.createElement('h3'); heading.textContent = value.name; panel.append(heading);
                paragraph(panel, `Stored value: ${value.stored} — ${value.exact ? 'exact' : 'rounded'}`);
                paragraph(panel, value.kind === 'zero' ? 'Zero: all exponent and fraction bits are zero; there is no implicit leading 1.' :
                  `(-1)^${value.sign} × (${value.significand})₂ × 2^${value.power}`);
                if(value.kind !== 'zero') paragraph(panel, `p = ${value.power}; biased exponent E = p + ${value.bias} = ${value.biased}.`);
                paragraph(panel, `Sign | Exponent | Fraction: ${value.sign} | ${value.exponent} | ${value.fraction}`);
              }
            });
          }
          model.on('change:result', draw); draw();
          return () => model.off('change:result', draw);
        }
        export default {render};
        """
        _css = """
        .conversion-board {font: 17px/1.5 system-ui,sans-serif; color:#22352d; background:#fff; padding:16px; border:1px solid #b6c9bf; border-radius:8px;}
        .conversion-board .controls,.conversion-board .examples {display:flex; flex-wrap:wrap; align-items:end; gap:10px; margin:12px 0;}
        .conversion-board label {display:flex; flex-direction:column; gap:4px;}
        .conversion-board input,.conversion-board select,.conversion-board button {font:inherit; color:inherit; background:transparent; border:1px solid #829e90; border-radius:4px; padding:6px 10px;}
        .conversion-board button,.conversion-board summary {cursor:pointer;}
        .conversion-board summary {font-weight:600; padding:12px 0;}
        .conversion-board details {border-top:1px solid #b6c9bf;}
        .conversion-board p {overflow-wrap:anywhere;}
        .conversion-board .table-wrap {overflow-x:auto;}
        .conversion-board table {border-collapse:collapse; font-variant-numeric:tabular-nums; width:100%;}
        .conversion-board th,.conversion-board td {padding:5px 10px; text-align:left; border-bottom:1px solid #b6c9bf;}
        .conversion-board :focus-visible {outline:3px solid #b37b16; outline-offset:2px;}
        @media(prefers-color-scheme:dark) {.conversion-board {color:#e0ece5; background:#19241f;} .conversion-board select option {background:#19241f;}}
        """

        def __init__(self):
            super().__init__()
            self.refresh()

        @traitlets.observe("text", "base")
        def refresh(self, change=None):
            try:
                self.result = conversion_steps(self.text, self.base)
            except ValueError as error:
                self.result = {"error": str(error)}

    return (ConversionBoard,)


@app.cell(hide_code=True)
def _(ConversionBoard, mo):
    board = mo.ui.anywidget(ConversionBoard())
    board
    return (board,)


if __name__ == "__main__":
    app.run()
