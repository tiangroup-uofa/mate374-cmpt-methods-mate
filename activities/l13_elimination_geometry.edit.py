# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget>=0.9", "traitlets>=5", "numpy>=2.0"]
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
def intro(mo):
    mo.md(r"""
    ## Three equations as three planes

    Each equation $a_{i1}x+a_{i2}y+a_{i3}z=b_i$ describes a plane in 3D. The solution of the system is the single point where all three planes meet. Row operations replace one equation by a combination of two equations, so each step changes one plane while every plane keeps passing through the solution point.

    Edit the coefficients below, then step through the elimination in the scene.
    """)
    return


@app.cell(hide_code=True)
def inputs(mo):
    A_input = mo.ui.matrix(
        [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]],
        min_value=-9.0,
        max_value=9.0,
        step=0.5,
        precision=1,
        row_labels=["eq 1", "eq 2", "eq 3"],
        column_labels=["x", "y", "z"],
        label="Coefficient matrix A",
        debounce=True,
    )
    b_input = mo.ui.matrix(
        [8.0, -11.0, -3.0],
        min_value=-20.0,
        max_value=20.0,
        step=0.5,
        precision=1,
        row_labels=["b1", "b2", "b3"],
        label="Right-hand side b",
        debounce=True,
    )
    mo.hstack([A_input, b_input], justify="start", gap=2)
    return A_input, b_input


@app.cell
def naive_lu_code(np):
    def naive_lu(A):
        """Gaussian elimination without row swaps. Returns L and U with A = L @ U."""
        U = np.array(A, dtype=float)
        n = len(U)
        L = np.eye(n)
        for k in range(n - 1):               # pivot column k
            for i in range(k + 1, n):        # rows below the pivot
                L[i, k] = U[i, k] / U[k, k]  # multiplier m_ik
                U[i, :] -= L[i, k] * U[k, :]
        return L, U

    def solve_lu(L, U, b):
        """Solve L c = b (forward), then U x = c (backward)."""
        n = len(b)
        c = np.zeros(n)
        for i in range(n):
            c[i] = b[i] - L[i, :i] @ c[:i]
        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
        return c, x

    return naive_lu, solve_lu


@app.cell(hide_code=True)
def stage_builder(np):
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    def fmt(v):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        s = "0" if s in ("-0", "") else s
        return s.replace("-", "−")

    def R(i):
        return f"R{i}".translate(SUB)

    def pt(x):
        return "(" + ", ".join(fmt(v) for v in x) + ")"

    VAR = ["x", "y", "z"]
    ORIENT_U = [
        "Plane 1 keeps a general orientation.",
        "Plane 2 has no x term, so it is parallel to the x axis.",
        "Plane 3 has only a z term, so it is horizontal: z is constant on it.",
    ]

    def build_stages(A, b, tol=1e-12):
        """Record every row operation of elimination and Gauss–Jordan as a scene stage."""
        M = np.column_stack([A, b]).astype(float)
        L = np.eye(3)
        stages = []

        def add(phase, title, note, target=(), lines=(), emphasis=(0, 1, 2), fresh=()):
            stages.append(dict(
                M=M.tolist(), L=L.tolist(), phase=phase, title=title, note=note,
                target=list(target), lines=[list(p) for p in lines],
                emphasis=list(emphasis), fresh=[list(p) for p in fresh],
            ))

        add("Start", "Original system [A | b]",
            "Each row of [A | b] is one plane. All three planes pass through the glowing solution point. "
            "Drag the scene to look along the line where two planes meet.")

        # Forward elimination: store each multiplier in L.
        for k in range(2):
            if abs(M[k, k]) < tol:
                return None, f"Naive elimination stops: the pivot in row {k + 1}, column {k + 1} is zero."
            for i in range(k + 1, 3):
                m = M[i, k] / M[k, k]
                M[i] -= m * M[k]
                M[i, k] = 0.0
                L[i, k] = m
                name = f"m{i + 1}{k + 1}".translate(SUB)
                add("Forward elimination",
                    f"{R(i + 1)} ← {R(i + 1)} − {name} {R(k + 1)},   {name} = {fmt(m)}",
                    f"Plane {i + 1} is replaced by plane {i + 1} − {name} × plane {k + 1}. Every plane in this "
                    f"family contains the dark line where planes {i + 1} and {k + 1} meet, so plane {i + 1} swings "
                    f"about that line until its {VAR[k]} coefficient is zero. The multiplier {name} is saved in L.",
                    target=[i], lines=[(i, k)], fresh=[(i, k)])
        if abs(M[2, 2]) < tol:
            return None, "Naive elimination stops: the last pivot is zero, so A is singular."

        U, c = M[:, :3].copy(), M[:, 3].copy()
        x = np.linalg.solve(U, c)
        add("U reached", "Upper-triangular system [U | c]",
            " ".join(ORIENT_U) + " The solution point has not moved.")

        # Back substitution uses the triangular planes one at a time.
        add("Back substitution", f"Plane 3 gives z = {fmt(x[2])}",
            f"The horizontal plane 3 fixes the height z = {fmt(x[2])} without reference to x or y.",
            emphasis=[2])
        add("Back substitution", f"Planes 2 and 3 give y = {fmt(x[1])}",
            f"Planes 2 and 3 are both parallel to the x axis, so they meet in a line parallel to the x axis. "
            f"Every point on that line has y = {fmt(x[1])} and z = {fmt(x[2])}. Only x is still unknown.",
            lines=[(1, 2)], emphasis=[1, 2])
        add("Back substitution", f"Plane 1 gives x = {fmt(x[0])}",
            f"Plane 1 crosses the line from the previous step at x = {fmt(x[0])}, "
            f"which completes the solution {pt(x)}.",
            lines=[(1, 2)])

        # Gauss–Jordan: clear the entries above each pivot.
        after = {
            (1, 2): "Plane 2 now contains only y, so it is the vertical plane y = constant.",
            (0, 2): "Plane 1 no longer contains z, so it is vertical, parallel to the z axis.",
            (0, 1): "Plane 1 now contains only x, so it is the vertical plane x = constant.",
        }
        for k in (2, 1):
            for i in reversed(range(k)):
                m = M[i, k] / M[k, k]
                M[i] -= m * M[k]
                M[i, k] = 0.0
                add("Gauss–Jordan",
                    f"{R(i + 1)} ← {R(i + 1)} − ({fmt(m)}) {R(k + 1)}",
                    f"Plane {i + 1} swings about its line of intersection with plane {k + 1} until its "
                    f"{VAR[k]} coefficient is zero. {after[(i, k)]}",
                    target=[i], lines=[(i, k)])

        d = np.diag(M[:, :3]).copy()
        M /= d[:, None]
        add("Gauss–Jordan", "Divide each row by its diagonal entry",
            f"Dividing an equation by a nonzero number changes its coefficients but leaves its plane in place. "
            f"The system now reads x = {fmt(x[0])}, y = {fmt(x[1])}, z = {fmt(x[2])}: three coordinate planes "
            f"meeting at the same point.",
            target=[0, 1, 2])
        return dict(stages=stages, L=L, U=U, c=c, x=x, view=best_view(stages)), None

    def best_view(stages):
        """Camera azimuth and elevation that show every plane obliquely, neither edge-on nor face-on."""
        n = np.array([row[:3] for st in stages for row in st["M"]], dtype=float)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        az, el = np.meshgrid(np.linspace(-np.pi, np.pi, 145), np.linspace(0.25, 0.65, 9))
        f = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
        cos = np.abs(f @ n.T)
        score = np.minimum(cos, 0.85 - cos).min(axis=-1)
        i = np.unravel_index(np.argmax(score), score.shape)
        return [float(az[i]), float(el[i])]

    return (build_stages,)


@app.cell(hide_code=True)
def scene_class(anywidget, traitlets):
    SCENE_ESM = r"""
    const COLORS = ["#2f6db5", "#d9822b", "#3a9e6f"];
    const CSS = `
    .ps { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
    .ps-wrap { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; }
    .ps-left { flex: 3 1 380px; min-width: 260px; }
    .ps-right { flex: 2 1 260px; min-width: 240px; }
    .ps canvas { width: 100%; display: block; touch-action: none; cursor: grab;
                 border-radius: 8px; background: rgba(127,127,127,0.07); }
    .ps-hint { font-size: 12px; opacity: 0.65; margin-top: 4px; }
    .ps-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
    .ps button { font: inherit; padding: 3px 10px; border-radius: 6px; cursor: pointer;
                 border: 1px solid rgba(127,127,127,0.5); background: rgba(127,127,127,0.08); color: inherit; }
    .ps button:disabled { opacity: 0.35; cursor: default; }
    .ps-count { font-size: 12px; opacity: 0.7; margin-left: 4px; }
    .ps-phase { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; }
    .ps-title { font-weight: 600; margin: 2px 0 6px; font-variant-numeric: tabular-nums; }
    .ps-note { margin: 0 0 10px; }
    .ps-mats { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
    .ps-cap { font-size: 12px; opacity: 0.75; margin-bottom: 2px; }
    .ps table { border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums;
                border-left: 2px solid; border-right: 2px solid; border-radius: 5px; }
    .ps td { padding: 1px 7px; text-align: right; min-width: 2.2em; transition: background 0.3s; }
    .ps td.aug { border-left: 1px dashed rgba(127,127,127,0.7); }
    .ps td.dot { min-width: 0; padding: 0 2px 0 4px; }
    .ps td.zero { opacity: 0.35; }
    .ps td.fresh { font-weight: 700; background: rgba(217,130,43,0.22); }
    .ps td.later { opacity: 0.25; }
    .ps .sw { display: inline-block; width: 9px; height: 9px; border-radius: 50%; }
    .ps-sliders label { display: grid; grid-template-columns: 2.2em 1fr 3.5em; gap: 8px;
                        align-items: center; margin: 2px 0; font-variant-numeric: tabular-nums; }
    .ps-steps { font-variant-numeric: tabular-nums; margin: 4px 0 8px; }
    .ps-steps div { margin: 1px 0; }
    .ps-checks { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; margin: 6px 0; }
    `;

    // ---------- small vector helpers ----------
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => Math.hypot(a[0], a[1], a[2]);
    const unit = (a) => mul(a, 1 / norm(a));
    const lerp = (a, b, t) => a + (b - a) * t;
    const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

    function fmt(v) {
      let s = (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
      if (s === "-0") s = "0";
      return s.replace("-", "−");
    }
    const h = (tag, cls, text) => {
      const e = document.createElement(tag);
      if (cls) e.className = cls;
      if (text !== undefined) e.textContent = text;
      return e;
    };
    const hexA = (hex, a) => {
      const n = parseInt(hex.slice(1), 16);
      return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
    };

    // Clip a convex polygon to the box |p - C| <= H in every coordinate.
    function clipBox(poly, C, H) {
      for (let ax = 0; ax < 3; ax++) {
        for (const sg of [1, -1]) {
          const out = [];
          for (let i = 0; i < poly.length; i++) {
            const a = poly[i], b = poly[(i + 1) % poly.length];
            const fa = sg * (a[ax] - C[ax]) - H, fb = sg * (b[ax] - C[ax]) - H;
            if (fa <= 0) out.push(a);
            if ((fa <= 0) !== (fb <= 0)) {
              const t = fa / (fa - fb);
              out.push([lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]);
            }
          }
          poly = out;
          if (poly.length < 3) return [];
        }
      }
      return poly;
    }

    // Segment of the line p0 + t d that lies inside the box.
    function clipLine(p0, d, C, H) {
      let t0 = -1e9, t1 = 1e9;
      for (let ax = 0; ax < 3; ax++) {
        const lo = C[ax] - H - p0[ax], hi = C[ax] + H - p0[ax];
        if (Math.abs(d[ax]) < 1e-12) { if (lo > 0 || hi < 0) return null; continue; }
        let a = lo / d[ax], b = hi / d[ax];
        if (a > b) [a, b] = [b, a];
        t0 = Math.max(t0, a); t1 = Math.min(t1, b);
      }
      return t1 > t0 ? [add(p0, mul(d, t0)), add(p0, mul(d, t1))] : null;
    }

    // Intersection line of two planes n1.p = d1 and n2.p = d2.
    function planeLine(r1, r2) {
      const n1 = r1.slice(0, 3), n2 = r2.slice(0, 3);
      const d = cross(n1, n2), dd = dot(d, d);
      if (dd < 1e-12) return null;
      const p0 = mul(add(mul(cross(n2, d), r1[3]), mul(cross(d, n1), r2[3])), 1 / dd);
      return [p0, d];
    }

    function makeScene(canvas, getState) {
      const ctx = canvas.getContext("2d");
      const v0 = getState().view || [-0.95, 0.42];
      const view = { az: v0[0], el: v0[1] };
      const home = { ...view };
      let W = 400, Hpx = 360, fg = "#222";

      function basis() {
        const ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el), se = Math.sin(view.el);
        return { right: [-sa, ca, 0], up: [-se * ca, -se * sa, ce], fwd: [ce * ca, ce * sa, se] };
      }

      function draw() {
        const st = getState();
        const { C, H } = st;
        const B = basis();
        const s = Math.min(W, Hpx) / (2 * H * 1.62);
        const cx = W / 2, cy = Hpx / 2;
        const proj = (p) => {
          const q = sub(p, C);
          return [cx + s * dot(q, B.right), cy - s * dot(q, B.up), dot(q, B.fwd)];
        };
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        ctx.clearRect(0, 0, W, Hpx);
        ctx.lineJoin = "round";

        // Box edges.
        const corners = [];
        for (const i of [-1, 1]) for (const j of [-1, 1]) for (const k of [-1, 1])
          corners.push(add(C, [i * H, j * H, k * H]));
        ctx.strokeStyle = hexA("#888888", 0.35); ctx.lineWidth = 1;
        for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) {
          const diff = sub(corners[a], corners[b]);
          if ([0, 1, 2].filter((k) => Math.abs(diff[k]) > 1e-9).length !== 1) continue;
          const pa = proj(corners[a]), pb = proj(corners[b]);
          ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
        }

        // Plane patches, split into cells so that crossing planes sort correctly.
        const cells = [];
        const outlines = [];
        const N = 14, R = H * 1.8;
        for (const pl of st.planes) {
          const n = pl.row.slice(0, 3);
          const nn = norm(n);
          if (nn < 1e-9) continue;
          const nh = mul(n, 1 / nn);
          const dist = (dot(n, C) - pl.row[3]) / nn;
          const pc = sub(C, mul(nh, dist));
          if (Math.abs(dist) > H * 1.75) continue;
          const helper = Math.abs(nh[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
          const u = unit(cross(nh, helper)), v = cross(nh, u);
          const P = (a, b) => add(pc, add(mul(u, a), mul(v, b)));
          for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
            const a0 = -R + (2 * R * i) / N, a1 = -R + (2 * R * (i + 1)) / N;
            const b0 = -R + (2 * R * j) / N, b1 = -R + (2 * R * (j + 1)) / N;
            const poly = clipBox([P(a0, b0), P(a1, b0), P(a1, b1), P(a0, b1)], C, H);
            if (!poly.length) continue;
            const pp = poly.map(proj);
            const depth = pp.reduce((acc, q) => acc + q[2], 0) / pp.length;
            cells.push({ pp, depth, fill: hexA(pl.color, pl.alpha) });
          }
          const edge = clipBox([P(-R, -R), P(R, -R), P(R, R), P(-R, R)], C, H);
          if (edge.length) outlines.push({ pp: edge.map(proj), pl });
        }
        cells.sort((a, b) => a.depth - b.depth);
        for (const c of cells) {
          ctx.beginPath();
          ctx.moveTo(c.pp[0][0], c.pp[0][1]);
          for (const q of c.pp.slice(1)) ctx.lineTo(q[0], q[1]);
          ctx.closePath();
          ctx.fillStyle = c.fill;
          ctx.fill();
        }
        for (const o of outlines) {
          ctx.beginPath();
          ctx.moveTo(o.pp[0][0], o.pp[0][1]);
          for (const q of o.pp.slice(1)) ctx.lineTo(q[0], q[1]);
          ctx.closePath();
          ctx.setLineDash(o.pl.dashed ? [5, 4] : []);
          ctx.strokeStyle = hexA(o.pl.color, Math.min(1, o.pl.alpha * 2.2 + 0.1));
          ctx.lineWidth = o.pl.bold ? 2.6 : 1.3;
          ctx.stroke();
        }
        ctx.setLineDash([]);

        // Highlighted intersection lines.
        for (const [r1, r2] of st.lines) {
          const L = planeLine(r1, r2);
          if (!L) continue;
          const seg = clipLine(L[0], L[1], C, H);
          if (!seg) continue;
          const a = proj(seg[0]), b = proj(seg[1]);
          ctx.strokeStyle = fg; ctx.globalAlpha = 0.85; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
          ctx.globalAlpha = 1;
        }

        // Earlier solution, for comparison.
        if (st.ghost) {
          const q = proj(st.ghost);
          ctx.strokeStyle = fg; ctx.globalAlpha = 0.6; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(q[0], q[1], 5, 0, 2 * Math.PI); ctx.stroke();
          ctx.globalAlpha = 1;
        }

        // Solution point.
        if (st.point) {
          const inside = [0, 1, 2].every((k) => Math.abs(st.point[k] - C[k]) <= H + 1e-9);
          const q = proj(st.point);
          if (inside) {
            const g = ctx.createRadialGradient(q[0], q[1], 0, q[0], q[1], 22);
            g.addColorStop(0, "rgba(255,214,70,0.95)");
            g.addColorStop(0.35, "rgba(255,190,40,0.55)");
            g.addColorStop(1, "rgba(255,190,40,0)");
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(q[0], q[1], 22, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = "#b00020";
            ctx.beginPath(); ctx.arc(q[0], q[1], 4.2, 0, 2 * Math.PI); ctx.fill();
          } else {
            ctx.fillStyle = fg; ctx.font = "12px system-ui, sans-serif";
            ctx.fillText("solution is outside the view box", 10, 18);
          }
        }

        // Axis triad.
        const o = [38, Hpx - 34];
        ctx.font = "12px system-ui, sans-serif"; ctx.lineWidth = 1.6;
        ["x", "y", "z"].forEach((name, k) => {
          const e = [0, 0, 0]; e[k] = 1;
          const dx = 24 * dot(e, B.right), dy = -24 * dot(e, B.up);
          ctx.strokeStyle = fg; ctx.fillStyle = fg;
          ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(o[0] + dx, o[1] + dy); ctx.stroke();
          ctx.fillText(name, o[0] + dx * 1.35 - 3, o[1] + dy * 1.35 + 4);
        });
      }

      function resize() {
        W = canvas.clientWidth || 400;
        Hpx = Math.round(Math.min(480, Math.max(300, W * 0.82)));
        canvas.style.height = Hpx + "px";
        canvas.width = Math.round(W * devicePixelRatio);
        canvas.height = Math.round(Hpx * devicePixelRatio);
        fg = getComputedStyle(canvas).color || "#222";
        draw();
      }

      let drag = null;
      canvas.addEventListener("pointerdown", (e) => {
        drag = { x: e.clientX, y: e.clientY, az: view.az, el: view.el };
        canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing";
      });
      canvas.addEventListener("pointermove", (e) => {
        if (!drag) return;
        view.az = drag.az - (e.clientX - drag.x) * 0.01;
        view.el = Math.max(-1.5, Math.min(1.5, drag.el + (e.clientY - drag.y) * 0.01));
        draw();
      });
      const end = () => { drag = null; canvas.style.cursor = "grab"; };
      canvas.addEventListener("pointerup", end);
      canvas.addEventListener("pointercancel", end);
      canvas.addEventListener("dblclick", () => { Object.assign(view, home); draw(); });
      new ResizeObserver(resize).observe(canvas);
      return { draw, resize };
    }

    function matrixTable(rows, opts = {}) {
      const t = h("table");
      rows.forEach((row, i) => {
        const tr = h("tr");
        if (opts.swatch) {
          const td = h("td", "dot");
          const sw = h("span", "sw"); sw.style.background = COLORS[i];
          td.append(sw); tr.append(td);
        }
        row.forEach((v, j) => {
          const td = h("td", "", fmt(v));
          if (opts.aug && j === row.length - 1) td.classList.add("aug");
          if (opts.cellClass) { const c = opts.cellClass(i, j, v); if (c) td.classList.add(c); }
          if (opts.targets && opts.targets.includes(i)) td.style.background = hexA(COLORS[i], 0.16);
          tr.append(td);
        });
        t.append(tr);
      });
      return t;
    }

    // ---------------- elimination mode ----------------
    function renderEliminate(model, left, right, canvas) {
      const stages = model.get("stages");
      const C = model.get("center"), H = model.get("half_width");
      let cur = Math.min(model.get("stage"), stages.length - 1);
      let rows = stages[cur].M.map((r) => r.slice());
      let anim = null, playing = false, timer = null;

      const state = () => {
        const S = stages[cur];
        const moving = anim ? anim.target : S.target;
        const emph = anim ? [0, 1, 2] : S.emphasis;
        const lines = (anim ? anim.lines : S.lines).map(([a, b]) => [rows[a], rows[b]]);
        return {
          C, H, view: model.get("view"), point: C, lines,
          planes: rows.map((row, i) => ({
            row, color: COLORS[i],
            alpha: emph.includes(i) ? (moving.includes(i) ? 0.5 : 0.36) : 0.08,
            bold: moving.includes(i), dashed: !emph.includes(i),
          })),
        };
      };
      const scene = makeScene(canvas, state);

      const bar = h("div", "ps-bar");
      const bFirst = h("button", "", "⏮"), bPrev = h("button", "", "◀ Back"),
            bNext = h("button", "", "Next ▶"), bPlay = h("button", "", "▶ Play");
      bFirst.title = "Back to the original system";
      const count = h("span", "ps-count");
      bar.append(bFirst, bPrev, bNext, bPlay, count);
      const phase = h("div", "ps-phase"), title = h("div", "ps-title"), note = h("p", "ps-note");
      const mats = h("div", "ps-mats");
      right.append(bar, phase, title, note, mats);

      function panel() {
        const S = stages[cur];
        count.textContent = `step ${cur} of ${stages.length - 1}`;
        phase.textContent = S.phase;
        title.textContent = S.title;
        note.textContent = S.note;
        bFirst.disabled = bPrev.disabled = cur === 0 || !!anim;
        bNext.disabled = cur === stages.length - 1 || !!anim;
        bPlay.textContent = playing ? "❚❚ Pause" : "▶ Play";
        mats.replaceChildren();
        const triangular = S.phase === "U reached" || S.phase === "Back substitution";
        const aug = h("div");
        aug.append(h("div", "ps-cap", cur === 0 ? "[A | b]" : triangular ? "[U | c]" : "Current [A | b]"));
        aug.append(matrixTable(S.M, {
          swatch: true, aug: true, targets: S.target,
          cellClass: (i, j, v) => (j < 3 && i !== j && Math.abs(v) < 1e-12 ? "zero" : ""),
        }));
        const Lbox = h("div");
        Lbox.append(h("div", "ps-cap", "L (stored multipliers)"));
        Lbox.append(matrixTable(S.L, {
          cellClass: (i, j) => {
            if (S.fresh.some(([a, b]) => a === i && b === j)) return "fresh";
            const filled = i <= j || stages.slice(0, cur + 1).some((s) => s.fresh.some(([a, b]) => a === i && b === j));
            return filled ? "" : "later";
          },
        }));
        mats.append(aug, Lbox);
      }

      function go(k, animate) {
        if (k < 0 || k >= stages.length || k === cur || anim) return;
        const from = stages[cur], to = stages[k];
        const step = Math.abs(k - cur) === 1;
        const active = k > cur ? to : from;
        const A0 = from.M, A1 = to.M;
        const finish = () => {
          anim = null; cur = k; rows = to.M.map((r) => r.slice());
          model.set("stage", cur); model.save_changes();
          panel(); scene.draw();
          if (playing) timer = setTimeout(() => (cur < stages.length - 1 ? go(cur + 1, true) : stopPlay()), 1400);
        };
        const changed = [0, 1, 2].filter((i) => A0[i].some((v, j) => Math.abs(v - A1[i][j]) > 1e-12));
        if (!(animate && step && changed.length)) { finish(); return; }
        anim = { target: active.target, lines: active.lines };
        panel();
        const t0 = performance.now(), dur = 1600;
        const frame = (now) => {
          const t = ease(Math.min(1, (now - t0) / dur));
          rows = A0.map((r, i) => r.map((v, j) => lerp(v, A1[i][j], t)));
          scene.draw();
          if (t < 1) requestAnimationFrame(frame); else finish();
        };
        requestAnimationFrame(frame);
      }
      function stopPlay() { playing = false; clearTimeout(timer); panel(); }

      bFirst.onclick = () => { stopPlay(); go(0, false); };
      bPrev.onclick = () => { stopPlay(); go(cur - 1, true); };
      bNext.onclick = () => { stopPlay(); go(cur + 1, true); };
      bPlay.onclick = () => {
        if (playing) { stopPlay(); return; }
        playing = true;
        if (cur === stages.length - 1) { go(0, false); } else { panel(); go(cur + 1, true); }
      };
      panel();
    }

    // ---------------- LU re-solve mode ----------------
    function renderResolve(model, left, right, canvas) {
      const lu = model.get("lu");
      const { A, L, U, b0 } = lu;
      const C = model.get("center"), H = model.get("half_width");
      let b = model.get("b").slice();
      const show = { orig: true, tri: true };

      const solve = () => {
        const c = [b[0], 0, 0];
        c[1] = b[1] - L[1][0] * c[0];
        c[2] = b[2] - L[2][0] * c[0] - L[2][1] * c[1];
        const x = [0, 0, 0];
        x[2] = c[2] / U[2][2];
        x[1] = (c[1] - U[1][2] * x[2]) / U[1][1];
        x[0] = (c[0] - U[0][1] * x[1] - U[0][2] * x[2]) / U[0][0];
        return { c, x };
      };

      const state = () => {
        const { c, x } = solve();
        const planes = [];
        if (show.orig) A.forEach((r, i) => planes.push({ row: [...r, b[i]], color: COLORS[i], alpha: show.tri ? 0.1 : 0.34, dashed: show.tri }));
        if (show.tri) U.forEach((r, i) => planes.push({ row: [...r, c[i]], color: COLORS[i], alpha: 0.4, bold: i === 2 }));
        return { C, H, view: model.get("view"), point: x, ghost: C, lines: [], planes };
      };
      const scene = makeScene(canvas, state);

      const sliders = h("div", "ps-sliders");
      const inputs = b.map((v, i) => {
        const lab = h("label");
        const name = h("span"); name.innerHTML = `b<sub>${i + 1}</sub>`;
        const inp = h("input"); inp.type = "range";
        inp.min = b0[i] - 4; inp.max = b0[i] + 4; inp.step = 0.25; inp.value = v;
        const out = h("span", "", fmt(v));
        inp.oninput = () => { b[i] = parseFloat(inp.value); out.textContent = fmt(b[i]); update(); };
        lab.append(name, inp, out); sliders.append(lab);
        return [inp, out];
      });
      const reset = h("button", "", "Reset b");
      reset.onclick = () => { b = b0.slice(); inputs.forEach(([inp, out], i) => { inp.value = b[i]; out.textContent = fmt(b[i]); }); update(); };

      const checks = h("div", "ps-checks");
      [["orig", "original planes [A | b]"], ["tri", "triangular planes [U | c]"]].forEach(([key, text]) => {
        const lab = h("label"); const cb = h("input"); cb.type = "checkbox"; cb.checked = show[key];
        cb.onchange = () => { show[key] = cb.checked; scene.draw(); };
        lab.append(cb, " " + text); checks.append(lab);
      });

      const steps = h("div", "ps-steps");
      const mats = h("div", "ps-mats");
      const Lbox = h("div"); Lbox.append(h("div", "ps-cap", "L (saved)"), matrixTable(L));
      const Ubox = h("div"); Ubox.append(h("div", "ps-cap", "U (saved)"), matrixTable(U, { swatch: true, cellClass: (i, j, v) => (i > j ? "zero" : "") }));
      mats.append(Lbox, Ubox);
      right.append(sliders, reset, checks, steps, mats);

      const sub2 = (s) => s.replace(/(\d)(\d)/, "<sub>$1$2</sub>");
      function update() {
        const { c, x } = solve();
        const m = (i, j) => fmt(L[i][j]);
        steps.innerHTML = `
          <div class="ps-cap">Forward substitution with L gives the offsets c</div>
          <div>c<sub>1</sub> = b<sub>1</sub> = ${fmt(c[0])}</div>
          <div>c<sub>2</sub> = b<sub>2</sub> − (${m(1, 0)}) c<sub>1</sub> = ${fmt(c[1])}</div>
          <div>c<sub>3</sub> = b<sub>3</sub> − (${m(2, 0)}) c<sub>1</sub> − (${m(2, 1)}) c<sub>2</sub> = ${fmt(c[2])}</div>
          <div class="ps-cap" style="margin-top:6px">Back substitution with U gives the point</div>
          <div>z = c<sub>3</sub> / u<sub>33</sub> = ${fmt(x[2])}</div>
          <div>y = (c<sub>2</sub> − u<sub>23</sub>z) / u<sub>22</sub> = ${fmt(x[1])}</div>
          <div>x = (c<sub>1</sub> − u<sub>12</sub>y − u<sub>13</sub>z) / u<sub>11</sub> = ${fmt(x[0])}</div>`;
        model.set("b", b.slice()); model.save_changes();
        scene.draw();
      }
      update();
    }

    function render({ model, el }) {
      el.classList.add("ps");
      const style = h("style"); style.textContent = CSS;
      const wrap = h("div", "ps-wrap"), left = h("div", "ps-left"), right = h("div", "ps-right");
      const canvas = h("canvas");
      left.append(canvas, h("div", "ps-hint", "Drag to rotate · double-click to reset the view"));
      wrap.append(left, right);
      el.append(style, wrap);
      if (model.get("mode") === "resolve") renderResolve(model, left, right, canvas);
      else renderEliminate(model, left, right, canvas);
    }
    export default { render };
    """

    class PlaneScene(anywidget.AnyWidget):
        _esm = SCENE_ESM
        mode = traitlets.Unicode("eliminate").tag(sync=True)
        stages = traitlets.List([]).tag(sync=True)
        stage = traitlets.Int(0).tag(sync=True)
        center = traitlets.List([0.0, 0.0, 0.0]).tag(sync=True)
        half_width = traitlets.Float(4.0).tag(sync=True)
        view = traitlets.List([-0.95, 0.42]).tag(sync=True)
        lu = traitlets.Dict({}).tag(sync=True)
        b = traitlets.List([]).tag(sync=True)

    return (PlaneScene,)


@app.cell(hide_code=True)
def compute(A_input, b_input, build_stages, mo, np):
    A = np.asarray(A_input.value, dtype=float)
    b = np.asarray(b_input.value, dtype=float).reshape(-1)
    result, problem = build_stages(A, b)
    mo.stop(problem is not None, mo.callout(mo.md(problem or ""), kind="warn"))
    return A, b, result


@app.cell(hide_code=True)
def elimination_scene(PlaneScene, mo, result):
    elimination = mo.ui.anywidget(PlaneScene(
        mode="eliminate",
        stages=result["stages"],
        center=result["x"].tolist(),
        half_width=4.0,
        view=result["view"],
    ))
    mo.vstack([
        mo.md("### Forward elimination, back substitution, and Gauss–Jordan"),
        elimination,
    ])
    return


@app.cell(hide_code=True)
def lu_intro(mo):
    mo.md(r"""
    ### Reusing L and U for a new right-hand side

    Changing $\mathbf{b}$ slides each original plane parallel to itself because the coefficients in $\mathbf{A}$, which set the plane orientations, stay fixed. The triangular planes of $\mathbf{U}$ keep their orientations for the same reason. Forward substitution with the saved multipliers in $\mathbf{L}$ computes how far each triangular plane must slide, and back substitution reads off the new intersection point. Drag the $b_i$ sliders and compare the new intersection point with the original one, marked by an open circle. Watch whether any plane turns.
    """)
    return


@app.cell(hide_code=True)
def resolve_scene(A, PlaneScene, b, mo, result):
    resolve = mo.ui.anywidget(PlaneScene(
        mode="resolve",
        lu=dict(A=A.tolist(), L=result["L"].tolist(), U=result["U"].tolist(), b0=b.tolist()),
        b=b.tolist(),
        center=result["x"].tolist(),
        half_width=4.0,
        view=result["view"],
    ))
    resolve
    return (resolve,)


@app.cell
def check_lu(A, naive_lu, np, resolve, solve_lu):
    L, U = naive_lu(A)
    b_new = np.array(resolve.value["b"], dtype=float)
    c, x = solve_lu(L, U, b_new)

    print("L =\n", L)
    print("U =\n", U)
    print("max |L @ U - A|         :", np.abs(L @ U - A).max())
    print("b                        :", b_new)
    print("x from L and U           :", x)
    print("x from np.linalg.solve   :", np.linalg.solve(A, b_new))
    return


if __name__ == "__main__":
    app.run()
