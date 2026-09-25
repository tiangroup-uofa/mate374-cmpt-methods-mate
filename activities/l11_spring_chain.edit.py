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
    K_MAX = 20.0
    _labels = ["wall"] + [f"{_i + 1}" for _i in range(_n)]
    # Node 0 is the wall. Only neighbouring nodes start connected; any other
    # entry stays 0, and dragging it adds a longer-range spring.
    _k0 = [[5.0 if abs(_i - _j) == 1 else 0.0 for _j in range(_n + 1)] for _i in range(_n + 1)]
    springs = mo.ui.matrix(
        _k0,
        min_value=0.0,
        max_value=K_MAX,
        step=0.5,
        precision=1,
        symmetric=True,
        disabled=[[_i == _j for _j in range(_n + 1)] for _i in range(_n + 1)],
        row_labels=_labels,
        column_labels=_labels,
        label="Spring constants k (eV/Å²)",
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
    return K_MAX, force, springs


@app.cell
def solve(force, np, springs):
    S = np.asarray(springs.value, dtype=float)
    F = np.asarray(force.value, dtype=float)

    # Each spring adds k to both diagonal entries and -k between its two atoms.
    # Removing the wall's row and column leaves the stiffness matrix of the atoms.
    K = (np.diag(S.sum(axis=1)) - S)[1:, 1:]
    solvable = np.linalg.matrix_rank(K) == len(F)
    u = np.linalg.solve(K, F) if solvable else None
    spring_list = [
        [_i, _j, float(S[_i, _j])]
        for _i in range(len(S))
        for _j in range(_i + 1, len(S))
        if S[_i, _j] > 0
    ]
    return F, K, solvable, spring_list, u


@app.cell(hide_code=True)
def chain_widget(anywidget, traitlets):
    class SpringChain(anywidget.AnyWidget):
        springs = traitlets.List([]).tag(sync=True)
        f = traitlets.List([]).tag(sync=True)
        u = traitlets.List([]).tag(sync=True)
        k_max = traitlets.Float(20.0).tag(sync=True)
        _esm = r"""
        const NS = "http://www.w3.org/2000/svg";
        const SUB = ["₀", "₁", "₂", "₃", "₄", "₅"];
        const SOFT = [242, 176, 67], STIFF = [106, 61, 154];

        function springColour(t) {
          const c = SOFT.map((s, i) => Math.round(s + t * (STIFF[i] - s)));
          return `rgb(${c.join(",")})`;
        }

        // Zigzag along any curve p(s), s in [0, 1], with straight leads at both ends.
        function zigzag(p, turns = 7, h = 9, lead = 0.07) {
          const pts = [p(0), p(lead)];
          for (let i = 0; i < 2 * turns; i++) {
            const s = lead + ((1 - 2 * lead) * (i + 0.5)) / (2 * turns);
            const [x, y] = p(s), [xa, ya] = p(s - 1e-3), [xb, yb] = p(s + 1e-3);
            const len = Math.hypot(xb - xa, yb - ya);
            const sign = i % 2 ? 1 : -1;
            pts.push([x - (sign * h * (yb - ya)) / len, y + (sign * h * (xb - xa)) / len]);
          }
          pts.push(p(1 - lead), p(1));
          return "M " + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ");
        }

        function render({ model, el }) {
          const svg = document.createElementNS(NS, "svg");
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
            const springs = model.get("springs"), f = model.get("f"), u = model.get("u");
            const kMax = model.get("k_max");
            const n = f.length, wall = 40, gap = 140, y = 95, r = 15;
            const solved = u.length === n;
            const disp = solved ? [0, ...u] : new Array(n + 1).fill(0);

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

            // Magnify displacements so the largest neighbour stretch stays readable.
            const rel = disp.slice(1).map((d, i) => Math.abs(d - disp[i]));
            const pxPerA = Math.min(300, (0.3 * gap) / Math.max(...rel, 1e-9));
            const rest = disp.map((_, i) => wall + gap * i);
            const pos = rest.map((x, i) => x + pxPerA * disp[i]);
            const deepest = Math.max(0, ...springs.map(([i, j]) => j - i));
            const height = deepest > 1 ? 200 + 30 * deepest : 190;
            const right = Math.max(...pos.slice(1).map((x, i) => Math.max(x, x + 70 * f[i]) + 60));
            const width = Math.max(660, right);
            svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

            add("line", { x1: wall, y1: y - 40, x2: wall, y2: y + 40, class: "ink", "stroke-width": 3 });
            for (let j = -40; j < 40; j += 10) {
              add("line", { x1: wall, y1: y + j, x2: wall - 10, y2: y + j + 10, class: "ink", "stroke-width": 1 });
            }
            for (let i = 1; i <= n; i++) add("circle", { cx: rest[i], cy: y, r, class: "ghost" });

            for (const [i, j, k] of springs) {
              const t = Math.min(1, k / kMax);
              const style = {
                fill: "none", stroke: springColour(t),
                "stroke-width": 1.5 + 3 * t, "stroke-linejoin": "round",
              };
              const name = i === 0 ? `kw${SUB[j]}` : `k${SUB[i]}${SUB[j]}`;
              if (j - i === 1) {
                const x0 = i ? pos[i] + r : wall, x1 = pos[j] - r;
                add("path", { ...style, d: zigzag((s) => [x0 + s * (x1 - x0), y]) });
                add("text", { x: (x0 + x1) / 2, y: y + 30, class: "label", "text-anchor": "middle" },
                    `${name} = ${k.toFixed(1)}`);
              } else {
                // Longer-range springs hang below the chain as arcs.
                const x0 = pos[i], x1 = pos[j], y0 = y + (i ? r : 0), depth = 25 + 30 * (j - i);
                const arc = (s) => [x0 + (x1 - x0) * (1 - Math.cos(Math.PI * s)) / 2,
                                    y0 + depth * Math.sin(Math.PI * s)];
                add("path", { ...style, d: zigzag(arc, 5 + 2 * (j - i), 7) });
                add("text", { x: (x0 + x1) / 2, y: y0 + depth + 22, class: "label", "text-anchor": "middle" },
                    `${name} = ${k.toFixed(1)}`);
              }
            }

            for (let i = 1; i <= n; i++) {
              add("circle", { cx: pos[i], cy: y, r, class: "atom" });
              add("text", { x: pos[i], y: y + 5, class: "atom-label", "text-anchor": "middle" }, `${i}`);
              add("text", { x: pos[i], y: y - 24, class: "label", "text-anchor": "middle" },
                  solved ? `u${SUB[i]} = ${disp[i].toFixed(3)} Å` : `u${SUB[i]} = ?`);
              if (Math.abs(f[i - 1]) > 1e-9) {
                const len = 70 * f[i - 1];
                add("line", {
                  x1: pos[i], y1: y - 50, x2: pos[i] + len, y2: y - 50,
                  stroke: "#2ca02c", "stroke-width": 3, "marker-end": "url(#force-head)",
                });
                add("text", { x: pos[i] + len / 2, y: y - 60, class: "force", "text-anchor": "middle" },
                    `F${SUB[i]} = ${f[i - 1].toFixed(1)}`);
              }
            }

            if (solved) {
              const bar = 0.1 * pxPerA;
              add("line", { x1: width - 40 - bar, y1: height - 12, x2: width - 40, y2: height - 12, class: "ink", "stroke-width": 2 });
              add("text", { x: width - 50 - bar, y: height - 8, class: "label", "text-anchor": "end" }, "0.1 Å displacement");
            } else {
              add("text", { x: width / 2, y: height - 10, class: "warn", "text-anchor": "middle" },
                  "No unique equilibrium: at least one atom has no path to the wall");
            }
          }

          draw();
          for (const name of ["springs", "f", "u", "k_max"]) model.on(`change:${name}`, draw);
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
        .spring-chain .warn { fill: #d62728; font: bold 13px sans-serif; }
        """

    return (SpringChain,)


@app.cell(hide_code=True)
def layout(F, K, K_MAX, SpringChain, force, mo, n_atoms, solvable, spring_list, springs, u):
    _view = mo.ui.anywidget(
        SpringChain(springs=spring_list, f=F.tolist(), u=u.tolist() if solvable else [], k_max=K_MAX)
    )
    _K_view = mo.ui.matrix(K.tolist(), disabled=True, precision=1, label="K (eV/Å²)")
    _u_view = mo.ui.matrix(
        u.tolist() if solvable else [0.0] * len(F), disabled=True, precision=3,
        row_labels=[f"u{_i + 1}" for _i in range(len(F))],
        label="u (Å)" if solvable else "u (Å): no unique solution",
    )
    mo.vstack(
        [
            mo.hstack([n_atoms, springs], justify="start", gap=2, align="center"),
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
