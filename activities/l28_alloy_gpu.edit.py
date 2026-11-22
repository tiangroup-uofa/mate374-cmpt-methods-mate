# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import anywidget
    import marimo as mo
    import traitlets

    return anywidget, mo, traitlets


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class AlloyGPU(anywidget.AnyWidget):
        """Metropolis A-B swaps on a periodic lattice, run in parallel by WebGL.

        Each fragment-shader pass updates one set of non-touching nearest-neighbour
        pairs at once. Sixteen passes (eight pair sets in each direction) make one
        sweep of 2 N^2 swap attempts. Energy = (number of unlike bonds) x epsilon.
        Adapted to a fixed-composition alloy from the liquid-vapor lattice demo by
        Vilas Winstein (https://vilas.us/simulations/liquidvapor/).
        """

        kT = traitlets.Float(1.0).tag(sync=True)
        fraction = traitlets.Float(0.5).tag(sync=True)
        size = traitlets.Int(256).tag(sync=True)
        f_ab = traitlets.Float(0.0).tag(sync=True)
        sweeps = traitlets.Int(0).tag(sync=True)
        _esm = r"""
const VERT = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

// One pass: every pair in one non-overlapping set proposes an A-B swap at once.
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

function control(label, html) {
  const wrap = document.createElement("label");
  wrap.className = "alloy-control";
  wrap.innerHTML = '<span>' + label + '</span>' + html;
  return wrap;
}

function render({ model, el }) {
  el.classList.add("alloy-widget");
  const bar = document.createElement("div");
  bar.className = "alloy-bar";
  const kT = control("kT / ε", '<input type="range" min="0.5" max="4" step="0.05" value="' + model.get("kT") + '"><output></output>');
  const frac = control("B fraction", '<input type="range" min="0.1" max="0.9" step="0.05" value="' + model.get("fraction") + '"><output></output>');
  const size = control("lattice", '<select><option>128</option><option>256</option><option>512</option></select>');
  const speed = control("sweeps per frame", '<select><option>1</option><option>4</option><option>16</option></select>');
  const pause = document.createElement("button");
  const reset = document.createElement("button");
  pause.textContent = "pause";
  reset.textContent = "reset";
  bar.append(kT, frac, size, speed, pause, reset);
  size.querySelector("select").value = String(model.get("size"));
  speed.querySelector("select").value = "4";
  const canvas = document.createElement("canvas");
  const readout = document.createElement("div");
  readout.className = "alloy-readout";
  el.append(bar, canvas, readout);

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

  let N = 0, tex = [], fbo = [], src = 0, sweeps = 0, frame = 0, running = true, raf = 0;
  let pixels = null, t0 = performance.now(), sweeps0 = 0, rate = 0, fAB = NaN;

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
    N = Number(size.querySelector("select").value);
    const c = Number(frac.querySelector("input").value);
    canvas.width = N; canvas.height = N;
    const data = new Uint8Array(N * N);
    const nB = Math.round(c * N * N);
    data.fill(255, 0, nB);
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
    src = 0; sweeps = 0; sweeps0 = 0; t0 = performance.now(); fAB = NaN;
  }

  function sweep(temperature) {
    gl.useProgram(step);
    gl.viewport(0, 0, N, N);
    gl.uniform1i(U.N, N);
    gl.uniform1f(U.kT, temperature);
    for (let axis = 0; axis < 2; axis++)
      for (let a = 0; a < 4; a++)
        for (let b = 0; b < 2; b++) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1 - src]);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, tex[src]);
          gl.uniform1i(U.S, 0);
          gl.uniform1i(U.axis, axis); gl.uniform1i(U.a, a); gl.uniform1i(U.b, b);
          gl.uniform1ui(U.seed, (Math.random() * 4294967295) >>> 0);
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
          src = 1 - src;
        }
    sweeps++;
  }

  function measure() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[src]);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let unlike = 0, nB = 0;
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const v = pixels[4 * (y * N + x)] > 127;
        nB += v;
        unlike += v !== (pixels[4 * (y * N + (x + 1) % N)] > 127);
        unlike += v !== (pixels[4 * (((y + 1) % N) * N + x)] > 127);
      }
    fAB = unlike / (2 * N * N);
    const now = performance.now();
    rate = (sweeps - sweeps0) * 2 * N * N / ((now - t0) / 1000);
    t0 = now; sweeps0 = sweeps;
    const c = nB / (N * N);
    model.set("f_ab", fAB); model.set("sweeps", sweeps); model.save_changes();
    readout.innerHTML = "sweeps <b>" + sweeps + "</b> · unlike-bond fraction <b>" + fAB.toFixed(3) +
      "</b> · random mixture " + (2 * c * (1 - c)).toFixed(3) +
      " · <b>" + (rate / 1e6).toFixed(1) + "</b> million swap attempts per second";
  }

  function draw() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, N, N);
    gl.useProgram(show);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex[src]);
    gl.uniform1i(gl.getUniformLocation(show, "S"), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function loop() {
    if (running) {
      const temperature = Number(kT.querySelector("input").value);
      const k = Number(speed.querySelector("select").value);
      for (let i = 0; i < k; i++) sweep(temperature);
      if (++frame % 15 === 0) measure();
    }
    draw();
    raf = requestAnimationFrame(loop);
  }

  const sync = () => {
    kT.querySelector("output").textContent = Number(kT.querySelector("input").value).toFixed(2);
    frac.querySelector("output").textContent = Number(frac.querySelector("input").value).toFixed(2);
    model.set("kT", Number(kT.querySelector("input").value)); model.save_changes();
  };
  kT.querySelector("input").addEventListener("input", sync);
  frac.querySelector("input").addEventListener("input", () => { sync(); init(); });
  size.querySelector("select").addEventListener("change", init);
  reset.addEventListener("click", init);
  pause.addEventListener("click", () => { running = !running; pause.textContent = running ? "pause" : "run"; });
  sync(); init(); loop();
  return () => cancelAnimationFrame(raf);
}

export default { render };

"""
        _css = r"""
.alloy-widget{font-family:Arial,sans-serif;font-size:14px;max-width:560px}
.alloy-bar{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-bottom:8px}
.alloy-control{display:flex;gap:6px;align-items:center}
.alloy-widget canvas{width:100%;max-width:512px;aspect-ratio:1;image-rendering:pixelated;display:block;border:1px solid #8a8f98}
.alloy-readout{margin-top:6px}
"""

    return (AlloyGPU,)


@app.cell(hide_code=True)
def _(AlloyGPU, mo):
    alloy = mo.ui.anywidget(AlloyGPU(kT=1.0, fraction=0.5, size=256))
    alloy
    return


if __name__ == "__main__":
    app.run()
