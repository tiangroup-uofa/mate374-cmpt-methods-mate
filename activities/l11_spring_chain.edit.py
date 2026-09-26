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
    _labels = [f"{_i}" for _i in range(_n + 1)]
    # Atom 0 is held fixed. Only neighbouring atoms start connected; any other
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
    # Atom 0 has u0 = 0, so removing its row and column leaves the system for atoms 1..n.
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
        const SOFT = [242, 176, 67], STIFF = [106, 61, 154];

        function springColour(t) {
          const c = SOFT.map((s, i) => Math.round(s + t * (STIFF[i] - s)));
          return `rgb(${c.join(",")})`;
        }

        // Longer springs run in their own lanes above and below the centre line.
        const LANES = [-12, 12, -18, 18];

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
            const n = f.length, x0Fixed = 30, gap = 110, y = 90, r = 23;
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
            const pxPerA = Math.min(300, (0.25 * gap) / Math.max(...rel, 1e-9));
            const rest = disp.map((_, i) => x0Fixed + gap * i);
            const pos = rest.map((x, i) => x + pxPerA * disp[i]);

            const near = springs.filter(([i, j]) => j - i === 1);
            const far = springs.filter(([i, j]) => j - i > 1)
              .sort((a, b) => (a[1] - a[0]) - (b[1] - b[0]) || a[0] - b[0]);
            const placed = [];
            for (const s of far) {
              const lane = LANES.find((dy) => !placed.some(
                ([p, q, , pdy]) => pdy === dy && p < s[1] && s[0] < q));
              placed.push([...s, lane ?? LANES[0]]);
            }

            const height = 160 + 16 * placed.length;
            const right = Math.max(...pos.slice(1).map((x, i) => Math.max(x, x + 60 * f[i]) + 40));
            const width = Math.max(420, right);
            svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
            svg.setAttribute("width", width);
            svg.setAttribute("height", height);

            for (let i = 1; i <= n; i++) add("circle", { cx: rest[i], cy: y, r, class: "ghost" });

            // Each spring is a straight bond; stiffer bonds are darker and thicker.
            const bond = (x0, x1, yb, k, maxWidth) => {
              const t = Math.min(1, k / kMax);
              add("line", { x1: x0, y1: yb, x2: x1, y2: yb, stroke: springColour(t),
                            "stroke-width": 2 + (maxWidth - 2) * t });
            };
            // Each spring leaves a ball where its lane meets the ball's edge.
            const edge = (dy) => Math.sqrt(r * r - dy * dy);

            placed.forEach(([i, j, k, dy], idx) => {
              const x0 = pos[i] + edge(dy), x1 = pos[j] - edge(dy);
              bond(x0, x1, y + dy, k, 5);
              add("text", { x: (pos[i] + pos[j]) / 2, y: y + r + 34 + 15 * idx, class: "label",
                            "text-anchor": "middle" },
                  k.toFixed(1));
            });
            for (const [i, j, k] of near) {
              const x0 = pos[i] + r, x1 = pos[j] - r;
              bond(x0, x1, y, k, 10);
              add("text", { x: (x0 + x1) / 2, y: y + r + 18, class: "label", "text-anchor": "middle" },
                  k.toFixed(1));
            }

            add("circle", { cx: pos[0], cy: y, r, class: "fixed-atom" });
            add("text", { x: pos[0], y: y + 6, class: "atom-label", "text-anchor": "middle" }, "0");
            add("text", { x: pos[0], y: y - r - 8, class: "label", "text-anchor": "middle" }, "fixed");

            for (let i = 1; i <= n; i++) {
              add("circle", { cx: pos[i], cy: y, r, class: "atom" });
              add("text", { x: pos[i], y: y + 6, class: "atom-label", "text-anchor": "middle" }, `${i}`);
              add("text", { x: pos[i], y: y - r - 8, class: "label", "text-anchor": "middle" },
                  solved ? disp[i].toFixed(3) : "?");
              if (Math.abs(f[i - 1]) > 1e-9) {
                const len = 60 * f[i - 1];
                add("line", {
                  x1: pos[i], y1: y - r - 28, x2: pos[i] + len, y2: y - r - 28,
                  stroke: "#2ca02c", "stroke-width": 3, "marker-end": "url(#force-head)",
                });
                add("text", { x: pos[i] + len / 2, y: y - r - 36, class: "force", "text-anchor": "middle" },
                    f[i - 1].toFixed(1));
              }
            }

            if (solved) {
              const bar = 0.1 * pxPerA;
              add("line", { x1: width - 40 - bar, y1: height - 12, x2: width - 40, y2: height - 12, class: "ink", "stroke-width": 2 });
              add("text", { x: width - 50 - bar, y: height - 8, class: "label", "text-anchor": "end" }, "0.1 Å displacement");
            } else {
              add("text", { x: width / 2, y: height - 10, class: "warn", "text-anchor": "middle" },
                  "No unique equilibrium: at least one atom has no spring path to atom 0");
            }
          }



          draw();
          for (const name of ["springs", "f", "u", "k_max"]) model.on(`change:${name}`, draw);
        }
        export default { render };
        """
        _css = r"""
        .spring-chain { max-width: 100%; height: auto; margin: 0 auto; display: block; color: inherit; }
        .spring-chain .ink { stroke: currentColor; }
        .spring-chain .ghost { fill: none; stroke: currentColor; stroke-opacity: 0.35; stroke-dasharray: 4 3; }
        .spring-chain .atom { fill: #1f77b4; stroke: currentColor; stroke-width: 1; }
        .spring-chain .fixed-atom { fill: #7f7f7f; stroke: currentColor; stroke-width: 2.5; }
        .spring-chain .atom-label { fill: white; font: bold 15px sans-serif; }
        .spring-chain .label { fill: currentColor; font: 13px sans-serif; }
        .spring-chain .force { fill: #2ca02c; font: bold 13px sans-serif; }
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
