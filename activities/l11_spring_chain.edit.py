# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget", "traitlets", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import anywidget
    import traitlets
    import numpy as np

    return anywidget, mo, np, traitlets


@app.cell(hide_code=True)
def atom_count(mo):
    n_atoms = mo.ui.slider(1, 4, value=3, step=1, label="Atoms", show_value=True)
    return (n_atoms,)


@app.cell(hide_code=True)
def inputs(mo, n_atoms):
    _n = n_atoms.value
    K_MIN, K_MAX = 1.0, 20.0
    spring_k = mo.ui.matrix(
        [[5.0] * _n],
        min_value=K_MIN,
        max_value=K_MAX,
        step=0.5,
        precision=1,
        column_labels=[f"k{_i + 1}" for _i in range(_n)],
        label="Spring constants (eV/Å²)",
    )
    force = mo.ui.matrix(
        [0.0] * (_n - 1) + [0.5],
        min_value=-1.0,
        max_value=1.0,
        step=0.1,
        precision=1,
        row_labels=[f"F{_i + 1}" for _i in range(_n)],
        label="F (eV/Å)",
    )
    return K_MAX, K_MIN, force, spring_k


@app.cell
def solve(force, np, spring_k):
    k = np.asarray(spring_k.value, dtype=float).ravel()
    F = np.asarray(force.value, dtype=float)

    # Spring 1 ties atom 1 to the wall; spring i ties atom i-1 to atom i.
    k_right = np.append(k[1:], 0.0)
    K = np.diag(k + k_right) - np.diag(k[1:], 1) - np.diag(k[1:], -1)
    u = np.linalg.solve(K, F)
    return F, K, k, u


@app.cell(hide_code=True)
def chain_widget(anywidget, traitlets):
    class SpringChain(anywidget.AnyWidget):
        k = traitlets.List([]).tag(sync=True)
        f = traitlets.List([]).tag(sync=True)
        u = traitlets.List([]).tag(sync=True)
        k_range = traitlets.List([1.0, 20.0]).tag(sync=True)
        _esm = r"""
        const NS = "http://www.w3.org/2000/svg";
        const SUB = ["₀", "₁", "₂", "₃", "₄", "₅"];
        const SOFT = [242, 176, 67], STIFF = [106, 61, 154];

        function springColour(t) {
          const c = SOFT.map((s, i) => Math.round(s + t * (STIFF[i] - s)));
          return `rgb(${c.join(",")})`;
        }

        function zigzag(x0, x1, y, turns = 7, h = 9) {
          const lead = 8, len = x1 - x0 - 2 * lead;
          let d = `M ${x0} ${y} L ${x0 + lead} ${y}`;
          for (let i = 0; i < 2 * turns; i++) {
            const x = x0 + lead + (len * (i + 0.5)) / (2 * turns);
            d += ` L ${x} ${y + (i % 2 ? h : -h)}`;
          }
          return d + ` L ${x1 - lead} ${y} L ${x1} ${y}`;
        }

        function render({ model, el }) {
          const svg = document.createElementNS(NS, "svg");
          svg.setAttribute("viewBox", "0 0 660 200");
          svg.setAttribute("class", "spring-chain");
          el.appendChild(svg);

          const add = (tag, attrs, text) => {
            const node = document.createElementNS(NS, tag);
            for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
            if (text !== undefined) node.textContent = text;
            svg.appendChild(node);
            return node;
          };

          function draw() {
            svg.replaceChildren();
            const k = model.get("k"), f = model.get("f"), u = model.get("u");
            const [kMin, kMax] = model.get("k_range");
            const n = u.length, wall = 40, gap = 140, y = 110, r = 15;

            const defs = add("defs", {});
            const marker = document.createElementNS(NS, "marker");
            for (const [key, value] of Object.entries({
              id: "force-head", viewBox: "0 0 10 10", refX: 8, refY: 5,
              markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse",
            })) marker.setAttribute(key, value);
            const head = document.createElementNS(NS, "path");
            head.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
            head.setAttribute("fill", "#2ca02c");
            marker.appendChild(head);
            defs.appendChild(marker);

            // Magnify displacements so the largest spring stretch stays readable.
            const rel = u.map((ui, i) => Math.abs(ui - (i ? u[i - 1] : 0)));
            const pxPerA = Math.min(300, (0.3 * gap) / Math.max(...rel, 1e-9));

            add("line", { x1: wall, y1: y - 40, x2: wall, y2: y + 40, class: "ink", "stroke-width": 3 });
            for (let j = -40; j < 40; j += 10) {
              add("line", { x1: wall, y1: y + j, x2: wall - 10, y2: y + j + 10, class: "ink", "stroke-width": 1 });
            }

            const rest = u.map((_, i) => wall + gap * (i + 1));
            const pos = rest.map((x, i) => x + pxPerA * u[i]);
            const right = Math.max(...pos.map((x, i) => Math.max(x, x + 70 * f[i]) + 60));
            const width = Math.max(660, right);
            svg.setAttribute("viewBox", `0 0 ${width} 200`);

            for (let i = 0; i < n; i++) {
              add("circle", { cx: rest[i], cy: y, r, class: "ghost" });
            }
            for (let i = 0; i < n; i++) {
              const x0 = i ? pos[i - 1] + r : wall, x1 = pos[i] - r;
              const t = Math.min(1, Math.max(0, (k[i] - kMin) / (kMax - kMin)));
              add("path", {
                d: zigzag(x0, x1, y), fill: "none", stroke: springColour(t),
                "stroke-width": 1.5 + 3 * t, "stroke-linejoin": "round",
              });
              add("text", { x: (x0 + x1) / 2, y: y + 30, class: "label", "text-anchor": "middle" },
                  `k${SUB[i + 1]} = ${k[i].toFixed(1)}`);
            }
            for (let i = 0; i < n; i++) {
              add("circle", { cx: pos[i], cy: y, r, class: "atom" });
              add("text", { x: pos[i], y: y + 5, class: "atom-label", "text-anchor": "middle" }, `${i + 1}`);
              add("text", { x: pos[i], y: y + 52, class: "label", "text-anchor": "middle" },
                  `u${SUB[i + 1]} = ${u[i].toFixed(3)} Å`);
              if (Math.abs(f[i]) > 1e-9) {
                const len = 70 * f[i];
                add("line", {
                  x1: pos[i], y1: y - 42, x2: pos[i] + len, y2: y - 42,
                  stroke: "#2ca02c", "stroke-width": 3, "marker-end": "url(#force-head)",
                });
                add("text", { x: pos[i] + len / 2, y: y - 52, class: "force", "text-anchor": "middle" },
                    `F${SUB[i + 1]} = ${f[i].toFixed(1)}`);
              }
            }

            const bar = 0.1 * pxPerA;
            add("line", { x1: width - 40 - bar, y1: 188, x2: width - 40, y2: 188, class: "ink", "stroke-width": 2 });
            add("text", { x: width - 50 - bar, y: 192, class: "label", "text-anchor": "end" }, "0.1 Å displacement");
          }

          draw();
          for (const name of ["k", "f", "u", "k_range"]) model.on(`change:${name}`, draw);
        }
        export default { render };
        """
        _css = r"""
        .spring-chain { width: 100%; max-width: 760px; display: block; color: inherit; }
        .spring-chain .ink { stroke: currentColor; }
        .spring-chain .ghost { fill: none; stroke: currentColor; stroke-opacity: 0.35; stroke-dasharray: 4 3; }
        .spring-chain .atom { fill: #1f77b4; stroke: currentColor; stroke-width: 1; }
        .spring-chain .atom-label { fill: white; font: bold 13px sans-serif; }
        .spring-chain .label { fill: currentColor; font: 12px sans-serif; }
        .spring-chain .force { fill: #2ca02c; font: bold 12px sans-serif; }
        """

    return (SpringChain,)


@app.cell(hide_code=True)
def layout(K, K_MAX, K_MIN, SpringChain, force, k, mo, n_atoms, spring_k, u):
    _view = mo.ui.anywidget(
        SpringChain(k=k.tolist(), f=[float(_v) for _v in force.value], u=u.tolist(),
                    k_range=[K_MIN, K_MAX])
    )
    _K_view = mo.ui.matrix(K.tolist(), disabled=True, precision=1, label="K (eV/Å²)")
    _u_view = mo.ui.matrix(
        u.tolist(), disabled=True, precision=3,
        row_labels=[f"u{_i + 1}" for _i in range(len(u))], label="u (Å)",
    )
    mo.vstack(
        [
            mo.hstack([n_atoms, spring_k], justify="start", gap=2, align="end"),
            _view,
            mo.hstack(
                [_K_view, _u_view, mo.md("## ="), force],
                justify="center", align="center", gap=1,
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
