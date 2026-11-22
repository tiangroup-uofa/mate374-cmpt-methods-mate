# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget", "traitlets", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import anywidget
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import traitlets

    return anywidget, mo, np, plt, traitlets


@app.cell(hide_code=True)
def _(np):
    # Exact square-lattice values for energy = epsilon x (number of unlike bonds).
    # As an Ising model the coupling is J = epsilon / 2, so k Tc = epsilon / ln(1 + sqrt 2).
    KTC = 1.0 / np.log(1.0 + np.sqrt(2.0))  # k Tc / epsilon = 1.1346
    K_B = 8.617333e-5  # eV/K

    def c_boundary(t):
        """Equilibrium B fraction of the A-rich phase at T/Tc = t < 1 (Yang 1952)."""
        t = np.asarray(t, dtype=float)
        x = np.sinh(1.0 / (KTC * t)) ** -4.0
        return (1.0 - np.clip(1.0 - x, 0.0, None) ** 0.125) / 2.0

    return KTC, K_B, c_boundary


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class AlloyGPU(anywidget.AnyWidget):
        """Metropolis A-B swaps on a periodic square lattice, run in parallel by WebGL.

        Each fragment-shader pass updates one set of non-touching nearest-neighbour
        pairs at once. Sixteen passes make one sweep of 2 N^2 swap attempts.
        Energy = epsilon x (number of unlike bonds). The traitlets carry inputs only.
        Adapted to a fixed-composition alloy from the liquid-vapour lattice demo by
        Vilas Winstein (https://vilas.us/simulations/liquidvapor/).
        """

        kT = traitlets.Float(0.8).tag(sync=True)  # k T / epsilon
        fraction = traitlets.Float(0.2).tag(sync=True)
        size = traitlets.Int(256).tag(sync=True)
        restart = traitlets.Int(0).tag(sync=True)
        running = traitlets.Bool(True).tag(sync=True)
        _esm = r"""
const VERT = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const STEP = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D S;
uniform int N, axis, a, b;
uniform float kT;
uniform uint seed;
out vec4 o;
float at(ivec2 q) { return texelFetch(S, (q + N) % N, 0).r; }
uint pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
void main() {
  ivec2 x = ivec2(gl_FragCoord.xy);
  ivec2 d = axis == 0 ? ivec2(1, 0) : ivec2(0, 1);   // pair direction
  ivec2 e = ivec2(d.y, d.x);                           // perpendicular
  int along = axis == 0 ? x.x : x.y;
  int across = axis == 0 ? x.y : x.x;
  float v = at(x);
  bool isP = (along % 4 == a) && (across % 2 == b);
  bool isQ = ((along - 1 + N) % 4 == a) && (across % 2 == b);
  if (!isP && !isQ) { o = vec4(v, 0.0, 0.0, 1.0); return; }
  ivec2 P = isP ? x : (x - d + N) % N;
  ivec2 Q = P + d;
  float vp = at(P), vq = at(Q);
  if (vp == vq) { o = vec4(v, 0.0, 0.0, 1.0); return; }
  float n1 = at(P - d), n2 = at(P + e), n3 = at(P - e);
  float m1 = at(Q + d), m2 = at(Q + e), m3 = at(Q - e);
  float before = abs(n1 - vp) + abs(n2 - vp) + abs(n3 - vp) + abs(m1 - vq) + abs(m2 - vq) + abs(m3 - vq);
  float after  = abs(n1 - vq) + abs(n2 - vq) + abs(n3 - vq) + abs(m1 - vp) + abs(m2 - vp) + abs(m3 - vp);
  float dE = after - before;                           // change in unlike bonds (units of epsilon)
  float u = float(pcg(pcg(uint(P.x) + uint(N) * uint(P.y)) ^ seed)) / 4294967296.0;
  bool accept = dE <= 0.0 || u < exp(-dE / kT);
  o = vec4(accept ? (isP ? vq : vp) : v, 0.0, 0.0, 1.0);
}`;

const SHOW = `#version 300 es
precision highp float;
uniform sampler2D S;
out vec4 o;
void main() {
  float v = texelFetch(S, ivec2(gl_FragCoord.xy), 0).r;
  o = v > 0.5 ? vec4(0.122, 0.467, 0.706, 1.0) : vec4(0.816, 0.824, 0.839, 1.0);
}`;

function compile(gl, vs, fs) {
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.bindAttribLocation(prog, 0, "p");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

function render({ model, el }) {
  el.classList.add("alloy-widget");
  const canvas = document.createElement("canvas");
  const readout = document.createElement("div");
  readout.className = "alloy-readout";
  el.append(canvas, readout);

  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
  if (!gl) { readout.textContent = "This browser does not provide WebGL 2."; return; }
  const step = compile(gl, VERT, STEP);
  const show = compile(gl, VERT, SHOW);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const U = {};
  for (const n of ["S", "N", "axis", "a", "b", "kT", "seed"]) U[n] = gl.getUniformLocation(step, n);
  const showS = gl.getUniformLocation(show, "S");

  let N = 0, tex = [], fbo = [], src = 0, sweeps = 0, raf = 0, alive = true;
  let perFrame = 1, lastFrame = performance.now(), lastMeasure = 0, t0 = 0, sweeps0 = 0;
  let pixels = null, label = null, stack = null;

  function texture(data) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, N, N, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    return t;
  }

  function init() {
    for (const t of tex) gl.deleteTexture(t);
    for (const f of fbo) gl.deleteFramebuffer(f);
    N = model.get("size");
    canvas.width = N; canvas.height = N;
    const data = new Uint8Array(N * N);
    data.fill(255, 0, Math.round(model.get("fraction") * N * N));
    for (let i = data.length - 1; i > 0; i--) {   // random arrangement, fixed composition
      const j = Math.floor(Math.random() * (i + 1));
      const t = data[i]; data[i] = data[j]; data[j] = t;
    }
    tex = [texture(data), texture(null)];
    fbo = tex.map((t) => {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      return f;
    });
    pixels = new Uint8Array(N * N * 4);
    label = new Int32Array(N * N);
    stack = new Int32Array(N * N);
    src = 0; sweeps = 0; sweeps0 = 0; perFrame = 1; t0 = performance.now(); lastMeasure = 0;
    measure();
  }

  function sweep() {
    gl.useProgram(step);
    gl.viewport(0, 0, N, N);
    gl.uniform1i(U.N, N);
    gl.uniform1f(U.kT, model.get("kT"));
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(U.S, 0);
    // Choose the 16 pair sets in random order. A fixed order would carry an
    // isolated atom across the lattice like a conveyor belt.
    for (let k = 0; k < 16; k++) {
          const r = Math.floor(Math.random() * 16);
          const axis = r >> 3, a = (r >> 1) & 3, b = r & 1;
          gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1 - src]);
          gl.bindTexture(gl.TEXTURE_2D, tex[src]);
          gl.uniform1i(U.axis, axis); gl.uniform1i(U.a, a); gl.uniform1i(U.b, b);
          gl.uniform1ui(U.seed, (Math.random() * 4294967295) >>> 0);
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
          src = 1 - src;
        }
    sweeps++;
  }

  // Read the lattice back, count unlike bonds and the clusters of the minority species.
  function measure() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[src]);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const M = N * N;
    let nB = 0, unlike = 0;
    for (let i = 0; i < M; i++) nB += pixels[4 * i] > 127;
    const minorityIsB = nB <= M / 2;
    const isM = (i) => (pixels[4 * i] > 127) === minorityIsB;
    label.fill(0);
    let large = 0, inLarge = 0, largest = 0, id = 0;
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const i = y * N + x, v = pixels[4 * i] > 127;
        unlike += v !== (pixels[4 * (y * N + (x + 1) % N)] > 127);
        unlike += v !== (pixels[4 * (((y + 1) % N) * N + x)] > 127);
        if (!isM(i) || label[i]) continue;
        id++; let top = 0, size = 0;
        stack[top++] = i; label[i] = id;
        while (top) {
          const k = stack[--top]; size++;
          const kx = k % N, ky = (k - kx) / N;
          for (const n of [ky * N + (kx + 1) % N, ky * N + (kx + N - 1) % N,
                           ((ky + 1) % N) * N + kx, ((ky + N - 1) % N) * N + kx])
            if (!label[n] && isM(n)) { label[n] = id; stack[top++] = n; }
        }
        if (size >= 20) { large++; inLarge += size; }
        if (size > largest) largest = size;
      }
    const nMin = minorityIsB ? nB : M - nB;
    const matrix = (nMin - inLarge) / (M - inLarge);
    const now = performance.now();
    const rate = (sweeps - sweeps0) / Math.max((now - t0) / 1000, 1e-3);
    t0 = now; sweeps0 = sweeps;
    readout.innerHTML =
      "<b>" + sweeps.toLocaleString() + "</b> sweeps · " + Math.round(rate).toLocaleString() + " sweeps/s" +
      "<br>unlike-bond fraction <b>" + (unlike / (2 * M)).toFixed(3) + "</b>" +
      " · particles ≥ 20 sites <b>" + large + "</b> (largest " + largest + ")" +
      " · minority fraction in the matrix <b>" + matrix.toFixed(3) + "</b>";
  }

  function draw() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, N, N);
    gl.useProgram(show);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex[src]);
    gl.uniform1i(showS, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function loop(now) {
    if (!alive) return;
    if (model.get("running")) {
      // Grow the batch while frames stay fast, shrink it when they slow down.
      const dt = now - lastFrame;
      if (dt < 25) perFrame = Math.min(perFrame * 1.25 + 1, 4096);
      else if (dt > 40) perFrame = Math.max(1, Math.floor(perFrame / 1.5));
      for (let i = 0; i < Math.floor(perFrame); i++) sweep();
      if (now - lastMeasure > 500) { measure(); lastMeasure = now; }
    }
    lastFrame = now;
    draw();
    raf = requestAnimationFrame(loop);
  }

  model.on("change:fraction", init);
  model.on("change:size", init);
  model.on("change:restart", init);
  init();
  raf = requestAnimationFrame(loop);
  return () => { alive = false; cancelAnimationFrame(raf); };
}

export default { render };
"""
        _css = r"""
.alloy-widget{font-family:Arial,sans-serif;font-size:13px;max-width:512px}
.alloy-widget canvas{width:100%;max-width:512px;aspect-ratio:1;image-rendering:pixelated;display:block;border:1px solid #8a8f98}
.alloy-readout{margin-top:6px;line-height:1.5}
"""

    return (AlloyGPU,)


@app.cell(hide_code=True)
def _(mo):
    PRESETS = {
        "Deep quench, c = 0.20": (0.70, 0.20),
        "Nucleation, c = 0.03": (0.70, 0.03),
        "Mixed above Tc, c = 0.20": (1.30, 0.20),
    }
    preset = mo.ui.dropdown(options=list(PRESETS), value="Deep quench, c = 0.20", label="Start from")
    return PRESETS, preset


@app.cell(hide_code=True)
def _(PRESETS, mo, preset):
    _t, _c = PRESETS[preset.value]
    t_slider = mo.ui.slider(0.40, 1.60, step=0.01, value=_t, label="T / Tc", show_value=True)
    c_slider = mo.ui.slider(0.02, 0.50, step=0.01, value=_c, label="B fraction", show_value=True)
    size = mo.ui.dropdown(options=["128", "256", "512"], value="256", label="lattice")
    eps = mo.ui.dropdown(options=["25", "50", "100"], value="50", label="ε (meV)")
    restart = mo.ui.button(label="restart from random", value=0, on_click=lambda v: v + 1)
    running = mo.ui.switch(value=True, label="run")
    return c_slider, eps, restart, running, size, t_slider


@app.cell(hide_code=True)
def _(AlloyGPU):
    lattice = AlloyGPU()
    return (lattice,)


@app.cell(hide_code=True)
def _(KTC, lattice, running, t_slider):
    # Temperature and run/pause act on the running lattice (a quench, not a restart).
    lattice.kT = KTC * t_slider.value
    lattice.running = running.value
    return


@app.cell(hide_code=True)
def _(c_slider, lattice, restart, size):
    # Composition, size, or the restart button start a new random lattice.
    lattice.fraction = c_slider.value
    lattice.size = int(size.value)
    lattice.restart = restart.value
    return


@app.cell(hide_code=True)
def _(KTC, K_B, c_boundary, c_slider, eps, mo, np, plt, t_slider):
    _t = np.r_[np.linspace(0.40, 0.999, 400), 1.0]
    _cb = c_boundary(_t)
    _cb[-1] = 0.5  # the two branches meet at the critical point
    fig, ax = plt.subplots(figsize=(3.0, 2.7), dpi=100)
    ax.fill_betweenx(_t, _cb, 1 - _cb, color="#dbe7f3")
    ax.plot(np.r_[_cb, (1 - _cb)[::-1]], np.r_[_t, _t[::-1]], color="#1f77b4", lw=1.6)
    ax.plot([c_slider.value], [t_slider.value], "o", color="#d62728", ms=7, zorder=3)
    ax.text(0.5, 0.62, "two phases", ha="center", fontsize=9)
    ax.text(0.5, 1.35, "one mixed phase", ha="center", fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0.4, 1.6), xlabel="B fraction", ylabel="T / Tc")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _tc_K = KTC * float(eps.value) * 1e-3 / K_B
    _T_K = t_slider.value * _tc_K
    if t_slider.value < 1:
        _ceq = float(c_boundary(t_slider.value))
        _where = "inside the two-phase region" if _ceq < c_slider.value < 1 - _ceq else "in the one-phase region"
        _eq = f"Equilibrium phases: {_ceq:.3f} and {1 - _ceq:.3f} B. The point is {_where}."
    else:
        _eq = "Above Tc the equilibrium state is one mixed phase at every composition."
    phase_panel = mo.hstack([
        fig,
        mo.md(f"k*T*/ε = {KTC * t_slider.value:.3f} · with ε = {eps.value} meV, "
              f"*T*c = {_tc_K:.0f} K and *T* = {_T_K:.0f} K  \n{_eq}  \n"
              f"Time: matching tracer diffusion to Cu at 1000 K gives about 22 µs per sweep, "
              f"so 10,000 sweeps ≈ 0.2 s."),
    ], widths=[1, 1.4], align="center", gap=1)
    plt.close(fig)
    return (phase_panel,)


@app.cell(hide_code=True)
def _(c_slider, eps, lattice, mo, phase_panel, preset, restart, running, size, t_slider):
    mo.vstack([
        mo.hstack([preset, t_slider, c_slider], justify="start", wrap=True),
        mo.hstack([size, eps, restart, running], justify="start", wrap=True),
        mo.ui.anywidget(lattice),
        phase_panel,
    ])
    return


if __name__ == "__main__":
    app.run()
