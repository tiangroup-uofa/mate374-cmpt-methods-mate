"""Four 'reading the pair curve' SVGs + pulled-pair GIF for L26."""
from pathlib import Path
import numpy as np
from l26_svg_figures import Panel, svg, text, line, INK, GREY, LGREY, RED, BLUE, GREEN, ORANGE, FS
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/"assets"
r0=2**(1/6); ri=(26/7)**(1/6)
U=lambda r:4*(r**-12-r**-6); U1=lambda r:24*(r**-7-2*r**-13); U2=lambda r:24*(26*r**-14-7*r**-8)
def atom(x,y,rad,c=BLUE):
    return (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{rad:.2f}" style="fill:url(#sh);stroke:{INK};stroke-width:0.25"/>')
SH='<radialGradient id="sh" cx="0.5" cy="0.5" fx="0.35" fy="0.35" r="0.6"><stop offset="0" style="stop-color:#ffffff"/><stop offset="1" style="stop-color:#1f77b4"/></radialGradient>'
def pair(p,r,y,lab=None,c=INK,broken=False):
    """Two atoms whose centre spacing equals r on the panel's x scale, centred on X(r)/2 style: left atom fixed at X(0.5)."""
    xl=p.X(r)-(p.X(r)-p.X(r-r))  # placeholder kept simple below
    sc=(p.X(1.5)-p.X(1.0))/0.5  # mm per sigma
    xr=p.X(r); xa=xr-sc*r*0.0
    return ""
def draw_pair(cx,y,r,sc,rad=2.6,bond=True,dash=False):
    x1,x2=cx-sc*r/2,cx+sc*r/2
    out=""
    if bond: out+=line([(x1,y),(x2,y)],GREY,0.5,"1,0.8" if dash else None)
    return out+atom(x1,y,rad)+atom(x2,y,rad)
W,H=104,62
def base(yr,yt,yl,cid,top=22,xr=(0.9,2.5)):
    p=Panel(20,top,W,H,xr,yr,cid); return p
def potential_fig():
    p=base((-1.3,1.2),[(-1,"−1"),(0,"0"),(1,"1")],"U / ε","a")
    r=np.linspace(0.9,2.5,1500); b=[p.axes([1,1.5,2,2.5],[(-1,"−1"),(0,"0"),(1,"1")],"","U / ε")]
    b.append(text(p.x0+W/2,p.y0+H+17.5,"Separation r / σ",anchor="middle"))
    b.append(p.curve(r,U(r),BLUE,0.9))
    b.append(line([(p.X(r0),p.Y(-1)),(p.X(r0),p.y0+H)],GREEN,0.35,"1.5,1"))
    b.append(f'<circle cx="{p.X(r0):.2f}" cy="{p.Y(-1):.2f}" r="1" style="fill:{GREEN}"/>')
    b.append(line([(p.X(r0),p.y0+H),(p.X(r0),p.y0+H+1.5)],GREEN,0.5))
    b.append(text(p.X(r0),p.y0+H+11,'r<tspan style="font-size:3px" dy="1">0</tspan>',anchor="middle",fill=GREEN))
    b.append(line([(p.X(r0),p.Y(-1)),(p.X(2.05),p.Y(-1))],GREY,0.3,"1.5,1"))
    b.append(line([(p.X(2.0),p.Y(-0.06)),(p.X(2.0),p.Y(-0.94))],RED,0.5,marker=True).replace("<path ","<path marker-start=\"url(#tri)\" "))
    b.append(text(p.X(2.05),p.Y(-0.45),"bond energy ε",fill=RED))
    b.append(text(p.X(r0)+1.5,p.Y(-1.2),"bond length r₀ = 2<tspan style=\"font-size:3px\" dy=\"-1.5\">1/6</tspan><tspan dy=\"1.5\"> σ</tspan>",fill=GREEN))
    b.append(draw_pair(p.X(r0)+4,11,r0,5.6,2.4))
    b.append(text(p.X(r0)+4+5.6*r0/2+5,12.4,"pair at rest at r₀",fill=GREY))
    open(A/"L26-curve-potential.svg","w").write(svg(W+26,H+44,"Bond energy and bond length on the pair potential","".join(b),p.clip()+SH))
def force_fig():
    p=base((-3.2,3.2),[(-2,"−2"),(0,"0"),(2,"2")],"","b",top=8)
    r=np.linspace(0.9,2.5,1500); b=[p.axes([1,1.5,2,2.5],[(-2,"−2"),(0,"0"),(2,"2")],"","U′ σ / ε")]
    yb=p.y0+H
    for rr,c in ((r0,GREEN),(ri,RED)):
        b.append(line([(p.X(rr),p.y0),(p.X(rr),yb+1.5)],c,0.35,"1.5,1"))
        b.append(line([(p.X(rr),yb),(p.X(rr),yb+1.5)],c,0.5))
    b.append(p.curve(r,U1(r),BLUE,0.9))
    b.append(f'<circle cx="{p.X(r0):.2f}" cy="{p.Y(0):.2f}" r="1" style="fill:{GREEN}"/>')
    b.append(f'<circle cx="{p.X(ri):.2f}" cy="{p.Y(2.396):.2f}" r="1" style="fill:{RED}"/>')
    # axis labels for the two special separations, on a row below the tick numbers
    sub=lambda s:f'<tspan style="font-size:3px" dy="1">{s}</tspan>'
    b.append(text(p.X(r0)-0.8,yb+11,"r"+sub("0")+'<tspan dy="-1"> ≈ 1.12</tspan>',anchor="end",fill=GREEN))
    b.append(text(p.X(ri)+0.8,yb+11,"r"+sub("infl")+'<tspan dy="-1"> ≈ 1.24</tspan>',anchor="start",fill=RED))
    b.append(text(p.x0+W/2,yb+17.5,"Separation r / σ",anchor="middle"))
    b.append(text(p.X(ri)+2,p.Y(2.75),"max restoring force ≈ 2.40 ε/σ at r"+sub("infl"),fill=RED))
    b.append(text(p.X(1.62),p.Y(1.55),"past r<tspan style=\"font-size:3px\" dy=\"1\">infl</tspan><tspan dy=\"-1\">, the restoring</tspan>",fill=INK))
    b.append(text(p.X(1.62),p.Y(1.55)+5,"force only weakens",fill=INK))
    b.append(text(p.X(1.3),p.Y(-1.2),"U′ = 0 at r"+sub("0")+'<tspan dy="-1">: no net force</tspan>',fill=GREEN))
    open(A/"L26-curve-force.svg","w").write(svg(W+26,H+28,"Maximum restoring force and the inflection point","".join(b),p.clip()))
def stiff_fig():
    p=base((-12,72),[(0,"0"),(30,"30"),(60,"60")],"","c",top=6)
    r=np.linspace(0.9,2.5,1500); b=[p.axes([1,1.5,2,2.5],[(0,"0"),(30,"30"),(60,"60")],"","U″ σ² / ε")]
    b.append(text(p.x0+W/2,p.y0+H+17.5,"Separation r / σ",anchor="middle"))
    b.append(line([(p.X(r0),p.y0),(p.X(r0),p.y0+H)],GREEN,0.35,"1.5,1"))
    b.append(line([(p.X(r0),p.y0+H),(p.X(r0),p.y0+H+1.5)],GREEN,0.5))
    b.append(text(p.X(r0),p.y0+H+11,'r<tspan style="font-size:3px" dy="1">0</tspan>',anchor="middle",fill=GREEN))
    b.append(p.curve(r,U2(r),BLUE,0.9))
    b.append(f'<circle cx="{p.X(r0):.2f}" cy="{p.Y(57.15):.2f}" r="1" style="fill:{GREEN}"/>')
    b.append(text(p.X(r0)+2.5,p.Y(64),"stiffness / spring constant",fill=GREEN))
    b.append(text(p.X(r0)+2.5,p.Y(64)+5.2,"k = U″(r<tspan style=\"font-size:3px\" dy=\"1\">0</tspan><tspan dy=\"-1\">) ≈ 57 ε/σ²</tspan>",fill=GREEN))
    # inset: U near r0 with the harmonic parabola -eps + 1/2 k (r-r0)^2
    PUR="#6a3d9a"
    q=Panel(p.X(1.62),p.y0+24,42,24,(r0-0.12,r0+0.14),(-1.05,-0.55),"ins")
    rr=np.linspace(r0-0.12,r0+0.14,300)
    b.append(f'<rect x="{q.x0-1:.2f}" y="{q.y0-1:.2f}" width="{q.w+2:.2f}" height="{q.h+2:.2f}" style="fill:white;stroke:{LGREY};stroke-width:0.3"/>')
    b.append(text(q.x0+q.w/2,q.y0-3,"curvature near the potential minimum",size=3.53,anchor="middle",fill=INK))
    b.append(q.curve(rr,U(rr),PUR,0.8)); b.append(q.curve(rr,-1+0.5*57.146*(rr-r0)**2,ORANGE,0.7,"1.5,0.8"))
    b.append(text(q.X(r0+0.08),q.Y(-0.99),"U(r)",size=3.53,fill=PUR))
    b.append(text(q.X(r0+0.105)-1.5,q.Y(-0.60)+1.2,"harmonic approx.",size=3.53,anchor="end",fill=ORANGE))
    open(A/"L26-curve-stiffness.svg","w").write(svg(W+26,H+26,"Bond stiffness as the curvature at the minimum","".join(b),p.clip()+q.clip()))
def thermal_fig():
    p=base((-1.15,0.15),[],"","d",top=6,xr=(0.95,2.2))
    r=np.linspace(0.95,2.2,1500)
    b=[p.axes([1,1.5,2],[(-1,"−1"),(0,"0")],"","U / ε")]
    yb=p.y0+H
    b.append(line([(p.X(r0),p.Y(-1)),(p.X(r0),yb+1.5)],GREEN,0.35,"1.5,1"))
    b.append(line([(p.X(r0),yb),(p.X(r0),yb+1.5)],GREEN,0.5))
    b.append(text(p.X(r0),yb+11,'r<tspan style="font-size:3px" dy="1">0</tspan>',anchor="middle",fill=GREEN))
    b.append(text(p.x0+W/2,yb+17.5,"Separation r / σ",anchor="middle"))
    b.append(p.curve(r,U(r),BLUE,0.9))
    from scipy.optimize import brentq
    mids=[]
    for E in (-0.9,-0.7,-0.5,-0.3):
        lo=brentq(lambda x:U(x)-E,0.95,r0); hi=brentq(lambda x:U(x)-E,r0,3.5)
        b.append(line([(p.X(lo),p.Y(E)),(p.X(hi),p.Y(E))],GREY,0.35))
        m=(lo+hi)/2; mids.append((p.X(m),p.Y(E)))
        b.append(f'<circle cx="{p.X(m):.2f}" cy="{p.Y(E):.2f}" r="0.8" style="fill:{RED}"/>')
        b.append(text(p.X(hi)+1.5,p.Y(E)+1.3,f"+{E+1:.1f} ε",size=3.53,fill=GREY))
    b.append(line([(p.X(r0),p.Y(-1))]+mids,RED,0.5,"1.2,0.8"))
    tx=p.X(1.62)
    b.append(text(tx,p.Y(-0.45),"line midpoints: mean",fill=RED))
    b.append(text(tx,p.Y(-0.45)+5,"separation grows with",fill=RED))
    b.append(text(tx,p.Y(-0.45)+10,"energy, so α &gt; 0",fill=RED))
    b.append(text(tx,p.Y(-0.98)-6,"+ε labels: energy above the minimum",size=3.2,fill=GREY))
    b.append(text(tx,p.Y(-0.98)+1,"melting: k<tspan style=\"font-size:3px\" dy=\"1\">B</tspan><tspan dy=\"-1\">T</tspan><tspan style=\"font-size:3px\" dy=\"1\">m</tspan><tspan dy=\"-1\"> ≈ 0.7 ε</tspan>",fill=INK))
    open(A/"L26-curve-thermal.svg","w").write(svg(W+26,H+26,"Thermal expansion and melting from the asymmetric well","".join(b),p.clip()))
if __name__=="__main__":
    potential_fig(); force_fig(); stiff_fig(); thermal_fig(); print("saved")
