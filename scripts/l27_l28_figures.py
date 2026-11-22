"""L27-L28 schematics and annotated plots in the course SVG style.

Inkscape-compatible SVG in mm units, Arial, grayscale with one red accent and
one blue hue for B atoms. Run from the project root:

    uv run python scripts/l27_l28_figures.py

PNG previews are exported separately at 300 dpi (see export_png below).
"""
from pathlib import Path
import math
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

INK = "#231f20"
GREY = "#5e5e5e"
MID = "#8a8f98"
LIGHT = "#b3b3b3"
PALE = "#e4e4e4"
RED = "#b5473a"
RED_FILL = "#f5d9d9"
BLUE = "#1f77b4"
BLUE_FILL = "#cde5ec"
FS = 4.23   # 12 pt main labels
SM = 3.53   # 10 pt secondary labels
PANEL = 6.35

DEFS = f"""<defs>
<marker id="tri" orient="auto-start-reverse" markerWidth="1" markerHeight="1" viewBox="0 0 1 1" refX="0" refY="0" style="overflow:visible" preserveAspectRatio="xMidYMid">
<path transform="scale(0.5)" d="M 5.77,0 -2.88,5 V -5 Z" style="fill:context-stroke;fill-rule:evenodd;stroke:context-stroke;stroke-width:1pt" />
</marker>
<radialGradient id="atom-a" cx="0.5" cy="0.5" r="0.5" fx="0.35" fy="0.32"><stop offset="0" stop-color="#ffffff" /><stop offset="1" stop-color="{MID}" /></radialGradient>
<radialGradient id="atom-b" cx="0.5" cy="0.5" r="0.5" fx="0.35" fy="0.32"><stop offset="0" stop-color="#ffffff" /><stop offset="1" stop-color="{BLUE}" /></radialGradient>
</defs>"""


def svg(width, height, title, desc, body):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">\n'
        f"<title>{title}</title>\n<desc>{desc}</desc>\n{DEFS}\n"
        f'<rect width="{width}" height="{height}" style="fill:white" />\n{body}\n</svg>\n'
    )


def layer(ident, label, content):
    return f'<g id="{ident}" inkscape:groupmode="layer" inkscape:label="{label}">\n{content}\n</g>'


def text(x, y, s, size=FS, anchor="start", fill=INK, weight="normal", style="normal"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" style="font-family:Arial;font-size:{size}px;'
        f'font-weight:{weight};font-style:{style};fill:{fill};text-anchor:{anchor}">{s}</text>'
    )


def it(s):
    return f'<tspan style="font-style:italic">{s}</tspan>'


def sub(s, size=2.9):
    return f'<tspan style="font-size:{size}px" dy="1.0">{s}</tspan><tspan dy="-1.0"> </tspan>'


def sup(s, size=2.9):
    return f'<tspan style="font-size:{size}px" dy="-1.6">{s}</tspan><tspan dy="1.6">\u200a</tspan>'


def line(points, color=INK, width=0.265, dash=None, marker=False, start_marker=False):
    d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points)
    extra = f";stroke-dasharray:{dash}" if dash else ""
    m = ";marker-end:url(#tri)" if marker else ""
    s = ";marker-start:url(#tri)" if start_marker else ""
    return f'<path d="{d}" style="fill:none;stroke:{color};stroke-width:{width}{extra}{m}{s}" />'


def rect(x, y, w, h, fill="none", stroke=INK, sw=0.265, rx=0, dash=None):
    extra = f";stroke-dasharray:{dash}" if dash else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" ry="{rx}" '
        f'style="fill:{fill};stroke:{stroke};stroke-width:{sw}{extra}" />'
    )


def atom(x, y, kind, r=1.9):
    grad = "atom-b" if kind == 1 else "atom-a"
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" style="fill:url(#{grad});stroke:{INK};stroke-width:0.3" />'


def write(name, content):
    path = ASSETS / name
    path.write_text(content, encoding="utf-8")
    return path


def export_png(svg_path, width_mm):
    """Export a 300 dpi PNG with headless Chrome (1 mm = 3.7795 CSS px)."""
    import re
    src = svg_path.read_text(encoding="utf-8")
    height_mm = float(re.search(r'height="([\d.]+)mm"', src).group(1))
    css_w = math.ceil(width_mm * 96 / 25.4)
    css_h = math.ceil(height_mm * 96 / 25.4)
    scale = 300 / 96
    html = (
        "<html><head><style>html,body{margin:0;padding:0;background:white}"
        f"img{{display:block;width:{css_w}px;height:{css_h}px}}</style></head>"
        f'<body><img src="{svg_path.as_uri()}"></body></html>'
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html)
        png = svg_path.with_suffix(".png")
        subprocess.run(
            [
                "google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--allow-file-access-from-files",
                f"--force-device-scale-factor={scale}",
                f"--window-size={css_w},{css_h}",
                f"--screenshot={png}", page.as_uri(),
            ],
            check=True, capture_output=True, timeout=60,
        )
    subprocess.run(
        ["convert", str(png), "-units", "PixelsPerInch", "-density", "300", str(png)],
        check=True,
    )
    return png


# ---------------------------------------------------------------- L27 figures

def l27_time_scales():
    """Log time axis: MD reach against measured copper self-diffusion at 800 K."""
    W, H = 170, 74
    x0, x1 = 8.0, 162.0
    lo, hi = -15, 1
    def X(t):
        return x0 + (math.log10(t) - lo) / (hi - lo) * (x1 - x0)
    ya = 44.0
    b = []
    # MD window band
    b.append(rect(X(1e-15), ya - 9, X(1e-6) - X(1e-15), 9, fill=BLUE_FILL, stroke="none"))
    b.append(text(X(1e-6) - 1.5, ya - 3.2, "large MD run: 10" + sup("9") + " steps of 1 fs",
                  size=SM, anchor="end", fill=BLUE))
    # axis
    b.append(line([(x0, ya), (x1 + 3, ya)], INK, 0.5, marker=True))
    for e in range(lo, hi + 1, 3):
        xx = X(10.0 ** e)
        b.append(line([(xx, ya), (xx, ya + 1.6)], INK, 0.35))
    labels = {-15: "1 fs", -12: "1 ps", -9: "1 ns", -6: "1 µs", -3: "1 ms", 0: "1 s"}
    for e, s in labels.items():
        b.append(text(X(10.0 ** e), ya + 5.6, s, size=SM, anchor="middle"))
    b.append(text(x1 + 3, ya + 5.6, "time", size=SM, anchor="end", style="italic"))
    # events above the axis
    events = [
        (168e-15, "one Cu–Cu vibration", "168 fs", INK),
        (1.76e-3, "one hop of a Cu atom", "about 2 ms", RED),
        (0.67, "rearrangement over 5 nm", "about 0.7 s", RED),
    ]
    heights = [24, 13, 24]
    for (t, name, val, col), hgt in zip(events, heights):
        xx = X(t)
        b.append(line([(xx, ya), (xx, ya - hgt - 0.8)], col, 0.35, dash="1.2,0.8"))
        b.append(f'<circle cx="{xx:.2f}" cy="{ya:.2f}" r="1.1" style="fill:{col};stroke:none" />')
        anchor = "middle" if t < 1e-6 else "end"
        dx = 1.6 if t > 0.1 else (-1.5 if t > 1e-6 else 0)
        if t > 1e-6 and t < 0.1:
            hgt = hgt - 4.0
        b.append(text(xx + dx, ya - hgt - 6.0, name, size=SM, anchor=anchor, fill=col))
        b.append(text(xx + dx, ya - hgt - 2.0, val, size=SM, anchor=anchor, fill=col, weight="bold"))
    # gap bracket below
    yb = ya + 13
    b.append(line([(X(1e-6), yb), (X(0.67), yb)], RED, 0.5, marker=True, start_marker=True))
    b.append(text((X(1e-6) + X(0.67)) / 2, yb + 5.4, "a factor of about 10" + sup("6"), size=FS, anchor="middle", fill=RED))
    b.append(text(X(1e-15), yb + 5.4, "Cu self-diffusion at 800 K:", size=SM))
    b.append(text(X(1e-15), yb + 10.0, it("D") + " ≈ 6 × 10" + sup("−18") + " m" + sup("2") + "/s", size=SM))
    return svg(W, H, "Time scales of MD and of atomic rearrangement in copper at 800 K",
               "Logarithmic time axis from 1 fs to 10 s. The MD band ends at one microsecond. One copper hop takes about 2 ms, and rearrangement over 5 nm about 0.7 s, from the measured self-diffusion coefficient.",
               layer("axis", "Time axis and events", "\n".join(b)))


def l27_two_state():
    """Two-state Metropolis check at kT = epsilon."""
    W, H = 150, 58
    b = []
    xl, xr = 8, 52
    y0, y1 = 46, 16
    b.append(line([(xl, y0), (xr, y0)], INK, 0.6))
    b.append(line([(xl, y1), (xr, y1)], INK, 0.6))
    b.append(text(xl - 1.5, y0 + 1.4, "0", size=FS, anchor="end"))
    b.append(text(xl - 1.5, y1 + 1.4, it("ε"), size=FS, anchor="end"))
    b.append(text(xr + 1.5, y0 + 1.4, "state 0", size=SM))
    b.append(text(xr + 1.5, y1 + 1.4, "state 1", size=SM))
    b.append(line([(22, y0 - 1.2), (22, y1 + 3.0)], RED, 0.5, marker=True))
    b.append(text(20.5, (y0 + y1) / 2 - 1.5, "accept", size=SM, anchor="end", fill=RED))
    b.append(text(20.5, (y0 + y1) / 2 + 3.0, it("e") + sup("−1") + " = 0.368", size=SM, anchor="end", fill=RED))
    b.append(line([(38, y1 + 1.2), (38, y0 - 3.0)], INK, 0.5, marker=True))
    b.append(text(39.5, (y0 + y1) / 2 - 1.5, "accept", size=SM))
    b.append(text(39.5, (y0 + y1) / 2 + 3.0, "always", size=SM))
    b.append(text(xl, 7.5, "a", size=PANEL, weight="bold"))
    # occupancy bars
    bx, by, bw, bh = 98, 10, 14, 36
    b.append(text(84, 7.5, "b", size=PANEL, weight="bold"))
    for k, (p, lab, col, fill) in enumerate([(0.731, "state 0", INK, PALE), (0.269, "state 1", RED, RED_FILL)]):
        x = bx + k * (bw + 12)
        h = p * bh
        b.append(rect(x, by + bh - h, bw, h, fill=fill, stroke=col, sw=0.35))
        b.append(text(x + bw / 2, by + bh - h - 1.6, f"{p:.3f}", size=SM, anchor="middle", fill=col, weight="bold"))
        b.append(text(x + bw / 2, by + bh + 4.6, lab, size=SM, anchor="middle", fill=col))
    b.append(line([(bx - 3, by + bh), (bx + 2 * bw + 15, by + bh)], INK, 0.5))
    b.append(text(bx - 4.5, by + bh / 2, "fraction of", size=SM, anchor="end"))
    b.append(text(bx - 4.5, by + bh / 2 + 4.4, "records", size=SM, anchor="end"))
    return svg(W, H, "Two-state check of the Metropolis rule",
               "Panel a: two energy levels 0 and epsilon, uphill acceptance exp(-1) and downhill acceptance 1 at kT equal to epsilon. Panel b: equilibrium fractions of records, 0.731 and 0.269.",
               layer("two-state", "Two-state Metropolis check", "\n".join(b)))


# ---------------------------------------------------------------- L28 figures

def lattice_panel(s, x0, y0, d, label, count_text):
    """Periodic L x L lattice; unlike bonds red, like bonds pale grey."""
    L = s.shape[0]
    out = []
    pos = lambda r, c: (x0 + c * d, y0 + r * d)
    for r in range(L):
        for c in range(L):
            x, y = pos(r, c)
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = (r + dr) % L, (c + dc) % L
                unlike = s[r, c] != s[rr, cc]
                col, w = (RED, 0.7) if unlike else (LIGHT, 0.3)
                if (dr, dc) == (0, 1) and c == L - 1:
                    out.append(line([(x, y), (x + d / 2, y)], col, w))
                    out.append(line([(x0 - d / 2, y), (x0, y)], col, w))
                elif (dr, dc) == (1, 0) and r == L - 1:
                    out.append(line([(x, y), (x, y + d / 2)], col, w))
                    out.append(line([(x, y0 - d / 2), (x, y0)], col, w))
                else:
                    x2, y2 = pos(rr, cc)
                    out.append(line([(x, y), (x2, y2)], col, w))
    for r in range(L):
        for c in range(L):
            out.append(atom(*pos(r, c), s[r, c], r=1.55))
    out.append(text(x0 - d / 2 - 1.0, y0 - d / 2 - 3.0, label, size=PANEL, weight="bold"))
    out.append(text(x0 + (L - 1) * d / 2, y0 + (L - 1) * d + d / 2 + 6.0, count_text, size=FS, anchor="middle", fill=RED))
    return "\n".join(out)


def unlike_count(s):
    return int(np.sum(s != np.roll(s, 1, 0)) + np.sum(s != np.roll(s, 1, 1)))


def l28_lattice():
    L = 8
    strips = np.zeros((L, L), int)
    strips[:, L // 2:] = 1
    rng = np.random.default_rng(374)
    mixed = np.array([0] * 32 + [1] * 32)
    rng.shuffle(mixed)
    mixed = mixed.reshape(L, L)
    n_mixed = unlike_count(mixed)
    d = 7.0
    W, H = 150, 76
    b = [
        lattice_panel(strips, 10, 12, d, "a", f"{it('N')}{sub('AB')}= {unlike_count(strips)}"),
        lattice_panel(mixed, 92, 12, d, "b", f"{it('N')}{sub('AB')}= {n_mixed}"),
    ]
    b.append(text(75, 30, "A", size=FS, anchor="middle"))
    b.append(atom(75, 35, 0))
    b.append(text(75, 45, "B", size=FS, anchor="middle"))
    b.append(atom(75, 50, 1))
    return svg(W, H, "Unlike bonds in two arrangements of 32 A and 32 B atoms",
               f"Periodic 8 by 8 lattices with 128 bonds. Panel a: two strips with 16 unlike bonds. Panel b: a random arrangement (seed 374) with {n_mixed} unlike bonds. Unlike bonds are red; bonds across the periodic boundary are drawn as half-length stubs at the edges.",
               layer("lattices", "Strip and random lattices", "\n".join(b))), n_mixed


def regular_binodal(kT_over_eps):
    """Outer root of ln(c/(1-c)) + H(1-2c) = 0 with H = 4 eps / kT, for c > 1/2."""
    H = 4.0 / kT_over_eps
    if H <= 2.0:
        return 0.5
    f = lambda c: math.log(c / (1 - c)) + H * (1 - 2 * c)
    a, b = 0.5 + 1e-9 + math.sqrt(max(0.0, 1 - 2 / H)) / 2, 1 - 1e-15
    for _ in range(200):
        m = (a + b) / 2
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    return m


def l28_three_answers():
    W, H = 120, 82
    x0, y0, w, h = 17, 6, 96, 62
    tmax = 2.25
    X = lambda c: x0 + c * w
    Y = lambda t: y0 + h - t / tmax * h
    b = []
    # regular-solution curve
    ts = np.linspace(0.05, 2.0, 400)
    right = [regular_binodal(t) for t in ts]
    pts = [(X(1 - c), Y(t)) for c, t in zip(right, ts)] + [(X(c), Y(t)) for c, t in zip(right[::-1], ts[::-1])]
    b.append('<path d="M' + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + f'" style="fill:{RED_FILL};stroke:{RED};stroke-width:0.6" />')
    # exact lattice curve (Onsager, Yang): m = (1 - sinh(eps/kT)^-4)^(1/8)
    tc = 1 / math.log(1 + math.sqrt(2))
    te = np.linspace(0.05, tc, 400)
    m = (1 - np.sinh(1 / te) ** -4) ** 0.125
    cl = (1 - m) / 2
    pts = [(X(c), Y(t)) for c, t in zip(cl, te)] + [(X(1 - c), Y(t)) for c, t in zip(cl[::-1], te[::-1])]
    b.append('<path d="M' + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + f'" style="fill:{BLUE_FILL};fill-opacity:0.85;stroke:{BLUE};stroke-width:0.6" />')
    # two-extremes estimate
    b.append(line([(X(0.5) - 9, Y(1 / math.log(2))), (X(0.5) + 9, Y(1 / math.log(2)))], GREY, 0.5, dash="1.5,1"))
    b.append(text(X(0.5), Y(1 / math.log(2)) + 4.8, "two extremes, 1.44", size=SM, anchor="middle", fill=GREY))
    # labels
    b.append(text(X(0.5), Y(2.0) - 1.8, "regular solution (A2), 2.00", size=SM, anchor="middle", fill=RED))
    b.append(text(X(0.5), Y(0.62), "exact lattice", size=SM, anchor="middle", fill=BLUE))
    b.append(text(X(0.5), Y(0.62) + 4.2, "(Onsager), 1.13", size=SM, anchor="middle", fill=BLUE))
    # H = 2.5 line and A2 roots
    t25 = 1.6
    b.append(line([(X(0), Y(t25)), (X(1), Y(t25))], GREY, 0.35, dash="1.2,0.8"))
    for c in (0.1448, 0.8552):
        b.append(f'<circle cx="{X(c):.2f}" cy="{Y(t25):.2f}" r="0.9" style="fill:{RED};stroke:none" />')
    b.append(text(X(0.1448) - 1.2, Y(t25) - 1.6, "0.145", size=SM, anchor="end", fill=RED))
    b.append(text(X(0.8552) + 1.2, Y(t25) - 1.6, "0.855", size=SM, fill=RED))
    # axes
    b.append(line([(x0, y0 - 2), (x0, y0 + h), (x0 + w + 3, y0 + h)], INK, 0.5))
    for c in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line([(X(c), y0 + h), (X(c), y0 + h + 1.3)], INK, 0.35))
        b.append(text(X(c), y0 + h + 5.0, f"{c:g}", size=SM, anchor="middle"))
    for t in (0, 0.5, 1.0, 1.5, 2.0):
        b.append(line([(x0 - 1.3, Y(t)), (x0, Y(t))], INK, 0.35))
        b.append(text(x0 - 2.0, Y(t) + 1.2, f"{t:.1f}", size=SM, anchor="end"))
    b.append(text(x0 + w / 2, y0 + h + 11.0, "fraction of B, " + it("c"), size=FS, anchor="middle"))
    b.append(f'<text transform="translate(4.6,{y0 + h / 2:.2f}) rotate(-90)" style="font-family:Arial;font-size:{FS}px;text-anchor:middle;fill:{INK}">'
             f'{it("k")}<tspan style="font-size:2.9px" dy="1">B</tspan><tspan dy="-1">{it("T")} / {it("ε")}</tspan></text>')
    return svg(W, H, "Separation temperatures of the same lattice alloy from three models",
               "Composition against kT/epsilon. Red region: regular-solution miscibility gap, peak at 2.00, with the A2 roots 0.145 and 0.855 at kT = 1.6 epsilon. Blue region: exact square-lattice coexistence from Onsager and Yang, peak at 1.13. Dashed grey: two-extremes estimate 1.44.",
               layer("phase-boundaries", "Three separation estimates", "\n".join(b)))


def l28_automaton():
    rows = ["BBAA", "BABA", "ABAB"]
    W, H = 104, 48
    d = 9.0
    x0 = 30
    b = []
    for k, row in enumerate(rows):
        y = 6 + k * 15
        b.append(text(x0 - 6, y + 6.0, f"{it('n')} = {k}", size=FS, anchor="end"))
        for i, ch in enumerate(row):
            fill = BLUE_FILL if ch == "B" else PALE
            col = BLUE if ch == "B" else INK
            b.append(rect(x0 + i * d, y, d, d, fill=fill, stroke=INK, sw=0.35))
            b.append(text(x0 + i * d + d / 2, y + d / 2 + 1.5, ch, size=FS, anchor="middle", fill=col, weight="bold"))
        if k < 2:
            for i in range(4):
                j = (i + 1) % 4
                if row[i] == "B" and row[j] == "A" and j == i + 1:
                    xa, xb = x0 + i * d + d / 2, x0 + j * d + d / 2
                    b.append(f'<path d="M{xa:.2f},{y + d + 0.6:.2f} Q{(xa + xb) / 2:.2f},{y + d + 4.6:.2f} {xb - 0.6:.2f},{y + d + 1.2:.2f}" style="fill:none;stroke:{RED};stroke-width:0.45;marker-end:url(#tri)" />')
    b.append(text(x0 + 4 * d + 3, 6 + d / 2 + 1.5, "B moves right", size=SM, fill=RED))
    b.append(text(x0 + 4 * d + 3, 6 + d / 2 + 5.8, "into an A cell", size=SM, fill=RED))
    b.append(text(x0 + 4 * d + 3, 36 + d / 2 + 1.5, "2 B in every row", size=SM))
    return svg(W, H, "A conservative cellular automaton on a periodic row",
               "Three rows of four cells: BBAA, BABA, ABAB. Red arrows mark each B that moves right into an A cell. Every row holds two B atoms.",
               layer("automaton", "Row sequence", "\n".join(b)))


def l28_free_energy():
    Hp = 2.5
    g = lambda c: Hp * c * (1 - c) + c * np.log(c) + (1 - c) * np.log(1 - c)
    W, H = 120, 70
    x0, y0, w, h = 17, 4, 98, 52
    glo, ghi = -0.20, 0.0
    X = lambda c: x0 + c * w
    Y = lambda v: y0 + (ghi - v) / (ghi - glo) * h
    b = []
    sp = (1 - math.sqrt(1 - 2 / Hp)) / 2
    b.append(rect(X(sp), y0, X(1 - sp) - X(sp), h, fill=RED_FILL, stroke="none"))
    b.append(text(X(0.5), y0 + 4.5, "spinodal region", size=SM, anchor="middle", fill=RED))
    cs = np.linspace(0.02, 0.98, 400)
    b.append(line([(X(c), Y(v)) for c, v in zip(cs, g(cs))], INK, 0.6))
    xa = 0.1448
    b.append(line([(X(0.05), Y(g(xa))), (X(0.95), Y(g(xa)))], GREY, 0.35, dash="1.5,1"))
    b.append(text(X(0.95), Y(g(xa)) + 4.2, "common tangent", size=SM, anchor="end", fill=GREY))
    for c in (xa, 1 - xa):
        b.append(f'<circle cx="{X(c):.2f}" cy="{Y(g(c)):.2f}" r="0.9" style="fill:{INK}" />')
    # split 0.4 / 0.6
    ga, gb = g(0.4), g(0.6)
    b.append(line([(X(0.4), Y(ga)), (X(0.6), Y(gb))], BLUE, 0.5))
    for c in (0.4, 0.6):
        b.append(f'<circle cx="{X(c):.2f}" cy="{Y(g(c)):.2f}" r="0.9" style="fill:{BLUE}" />')
    b.append(f'<circle cx="{X(0.5):.2f}" cy="{Y(g(0.5)):.2f}" r="0.9" style="fill:{INK}" />')
    b.append(text(X(0.5), Y(g(0.5)) - 3.0, "uniform 0.5", size=SM, anchor="middle"))
    b.append(text(X(0.5), Y(ga) + 5.2, "0.4 + 0.6: lower " + it("g"), size=SM, anchor="middle", fill=BLUE))
    # axes
    b.append(line([(x0, y0 - 1), (x0, y0 + h), (x0 + w + 2, y0 + h)], INK, 0.5))
    for c in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line([(X(c), y0 + h), (X(c), y0 + h + 1.3)], INK, 0.35))
        b.append(text(X(c), y0 + h + 5.0, f"{c:g}", size=SM, anchor="middle"))
    for v in (0.0, -0.1, -0.2):
        b.append(line([(x0 - 1.3, Y(v)), (x0, Y(v))], INK, 0.35))
        b.append(text(x0 - 2.0, Y(v) + 1.2, f"{v:.1f}".replace("-", "−"), size=SM, anchor="end"))
    b.append(text(x0 + w / 2, y0 + h + 11.0, "fraction of B, " + it("c"), size=FS, anchor="middle"))
    b.append(f'<text transform="translate(4.6,{y0 + h / 2:.2f}) rotate(-90)" style="font-family:Arial;font-size:{FS}px;text-anchor:middle;fill:{INK}">'
             f'{it("g")} / {it("k")}<tspan style="font-size:2.9px" dy="1">B</tspan><tspan dy="-1">{it("T")}</tspan></text>')
    return svg(W, H, "Regular-solution free energy at H = 2.5",
               "Free energy per site over kT against composition. The spinodal region between 0.276 and 0.724 is shaded. Splitting a uniform 0.5 alloy into 0.4 and 0.6 lowers the average free energy. The horizontal common tangent touches at 0.145 and 0.855.",
               layer("free-energy", "Regular-solution free energy", "\n".join(b)))


def l28_pair_sets():
    """Two of the sixteen passes: pairs updated together, and one pair's read neighbourhood."""
    L, d = 8, 6.4
    W, H = 128, 64
    b = []
    def panel(x0, y0, a, bb, label, show_read):
        out = []
        pos = lambda r, c: (x0 + c * d, y0 + r * d)
        pairs = [(r, c) for r in range(L) for c in range(L) if c % 4 == a and r % 2 == bb]
        if show_read:
            r, c = pairs[0] if a != 0 else pairs[5]
            reads = [(r, c - 1), (r, c + 2), (r - 1, c), (r + 1, c), (r - 1, c + 1), (r + 1, c + 1)]
            for rr, cc in reads:
                x, y = pos(rr % L, cc % L)
                out.append(rect(x - d / 2 + 0.3, y - d / 2 + 0.3, d - 0.6, d - 0.6, fill=PALE, stroke="none"))
        for r in range(L):
            for c in range(L):
                x, y = pos(r, c)
                out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="0.9" style="fill:{MID};stroke:none" />')
        for r, c in pairs:
            x, y = pos(r, c)
            out.append(rect(x - 1.9, y - 1.9, d + 3.8, 3.8, stroke=RED, sw=0.5, rx=1.9))
        out.append(text(x0 - d / 2 - 1.0, y0 - d / 2 - 2.6, label, size=PANEL, weight="bold"))
        return "\n".join(out)
    b.append(panel(6, 10, 0, 0, "a", True))
    b.append(panel(72, 10, 2, 1, "b", False))
    return svg(W, H, "Pairs updated together in one parallel pass",
               "Two 8 by 8 periodic lattices. Red rounded outlines mark the horizontal pairs proposed together in one pass: every fourth column and every second row. Panel a shades the six neighbours read by one pair. Panel b shows a different pass shifted by two columns and one row.",
               layer("pair-sets", "Parallel pair sets", b and "\n".join(b)))


if __name__ == "__main__":
    outputs = {
        "L27-time-scales.svg": (l27_time_scales(), 170),
        "L27-two-state.svg": (l27_two_state(), 150),
        "L28-three-answers.svg": (l28_three_answers(), 120),
        "L28-pair-sets.svg": (l28_pair_sets(), 128),
        "L28-free-energy.svg": (l28_free_energy(), 120),
    }
    lattice_svg, n_mixed = l28_lattice()
    outputs["L28-lattice-bonds.svg"] = (lattice_svg, 150)
    for name, (content, width) in outputs.items():
        path = write(name, content)
        export_png(path, width)
        print(name, "written")
    print("random lattice unlike bonds:", n_mixed)
