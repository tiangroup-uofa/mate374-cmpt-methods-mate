"""Hand-style SVG figures for L26 (mm units, Arial, grey structure + few hues)."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; ASSETS=ROOT/"assets"
INK,GREY,LGREY="#231f20","#5e5e5e","#8a8f98"
RED,BLUE,GREEN,ORANGE="#b5473a","#1f77b4","#2ca02c","#ff7f0e"
FS=4.23
MARK=('<marker id="tri" orient="auto-start-reverse" markerWidth="1" markerHeight="1" viewBox="0 0 1 1" '
      'refX="0" refY="0" style="overflow:visible" preserveAspectRatio="xMidYMid"><path transform="scale(0.5)" '
      'd="M 5.77,0 -2.88,5 V -5 Z" style="fill:context-stroke;fill-rule:evenodd;stroke:context-stroke;stroke-width:1pt"/></marker>')
def text(x,y,s,size=FS,anchor="start",fill=INK,weight="normal",style="normal",rotate=None):
    t=f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}"{t} style="font-family:Arial;font-size:{size}px;text-anchor:{anchor};'
            f'fill:{fill};font-weight:{weight};font-style:{style}">{s}</text>')
def line(pts,stroke=INK,w=0.5,dash=None,marker=False,fill="none"):
    d="M"+" L".join(f"{x:.3f},{y:.3f}" for x,y in pts)
    da=f";stroke-dasharray:{dash}" if dash else ""
    m=' marker-end="url(#tri)"' if marker else ""
    return f'<path d="{d}" style="fill:{fill};stroke:{stroke};stroke-width:{w}{da};stroke-linejoin:round"{m}/>'
class Panel:
    def __init__(s,x0,y0,w,h,xr,yr,cid):
        s.x0,s.y0,s.w,s.h,s.xr,s.yr,s.cid=x0,y0,w,h,xr,yr,cid
    def X(s,r): return s.x0+(r-s.xr[0])/(s.xr[1]-s.xr[0])*s.w
    def Y(s,v): return s.y0+s.h-(v-s.yr[0])/(s.yr[1]-s.yr[0])*s.h
    def clip(s): return f'<clipPath id="{s.cid}"><rect x="{s.x0}" y="{s.y0}" width="{s.w}" height="{s.h}"/></clipPath>'
    def curve(s,r,v,stroke,w=0.8,dash=None):
        v=np.clip(v,s.yr[0]-5,s.yr[1]+5)
        return f'<g clip-path="url(#{s.cid})">'+line(list(zip(s.X(r),s.Y(v))),stroke,w,dash)+'</g>'
    def axes(s,xticks,yticks,xlabel,ylabel,show_x=True,zero=True):
        out=[line([(s.x0,s.y0),(s.x0,s.y0+s.h),(s.x0+s.w,s.y0+s.h)],INK,0.5)]
        if zero and s.yr[0]<0<s.yr[1]: out.append(line([(s.x0,s.Y(0)),(s.x0+s.w,s.Y(0))],LGREY,0.3,"1.5,1"))
        for t in xticks:
            out.append(line([(s.X(t),s.y0+s.h),(s.X(t),s.y0+s.h+1.5)],INK,0.4))
            if show_x: out.append(text(s.X(t),s.y0+s.h+5.6,f"{t:g}",anchor="middle"))
        for t,lab in yticks:
            out.append(line([(s.x0,s.Y(t)),(s.x0-1.5,s.Y(t))],INK,0.4))
            out.append(text(s.x0-2.4,s.Y(t)+1.5,lab,anchor="end"))
        if show_x and xlabel: out.append(text(s.x0+s.w/2,s.y0+s.h+11.5,xlabel,anchor="middle"))
        out.append(text(s.x0-11,s.y0+s.h/2,ylabel,anchor="middle",rotate=-90))
        return "".join(out)
def svg(w,h,title,body,defs=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">\n<title>{title}</title>\n<defs>{MARK}{defs}</defs>\n'
            f'<rect width="{w}" height="{h}" style="fill:white"/>\n{body}\n</svg>\n')
def mie(r,n,m=6):
    c=n/(n-m)*(n/m)**(m/(n-m)); return c*(r**-n-r**-m)

def repulsion_family():
    p=Panel(18,4,100,60,(0.8,2.5),(-1.25,1.6),"c1")
    r=np.linspace(0.8,2.5,1600); rs=r[r>=1]
    b=[p.axes([1,1.5,2,2.5],[(-1,"−1"),(0,"0"),(1,"1")],"Separation r / σ","U / ε")]
    for n,dash in ((48,"1.2,0.8"),(24,"2.4,0.8")): b.append(p.curve(r,mie(r,n),LGREY,0.45,dash))
    b.append(p.curve(r,mie(r,12),RED,0.8))
    # hard sphere: wall at d = sigma, zero outside
    b.append(line([(p.X(1),p.y0),(p.X(1),p.Y(0)),(p.X(2.5),p.Y(0))],INK,1.1))
    # Sutherland: same wall, attractive -(sigma/r)^6 tail
    b.append(line([(p.X(1),p.y0)]+list(zip(p.X(rs),p.Y(-rs**-6))),BLUE,0.7))
    b.append(text(p.X(1.03),p.Y(1.42),"U → ∞",style="italic"))
    # n increases toward the vertical wall
    b.append(line([(p.X(0.83),p.Y(0.55)),(p.X(0.955),p.Y(0.55))],GREY,0.4,marker=True))
    b.append(text(p.X(0.82),p.Y(0.75),"n ↑",fill=GREY))
    # key, in curve colours, in the empty upper right
    ky=[(1.45,INK,1.1,None,"hard sphere: wall only"),(1.22,BLUE,0.7,None,"Sutherland: wall + attraction"),
        (0.99,LGREY,0.45,"2.4,0.8","n = 24, 48"),(0.76,RED,0.8,None,"Lennard–Jones: n = 12")]
    for y,c,w,d,lab in ky:
        b.append(line([(p.X(1.5),p.Y(y)),(p.X(1.62),p.Y(y))],c,w,d))
        b.append(text(p.X(1.66),p.Y(y)+1.4,lab,fill=c if c!=LGREY else GREY))
    b.append(text(p.X(1.68),p.Y(-0.55),"repulsion ∝ (σ/r)ⁿ",fill=GREY))
    b.append(text(p.X(1.68),p.Y(-0.55)+5,"n → ∞: hard wall",fill=GREY))
    open(ASSETS/"L26-repulsion-family.svg","w").write(svg(128,80,"Hard sphere, Sutherland, and Lennard–Jones potentials","".join(b),p.clip()))

def pair_three():
    r=np.linspace(0.95,2.3,1500); r0=2**(1/6); ri=(26/7)**(1/6)
    U=4*(r**-12-r**-6); U1=24*(r**-7-2*r**-13); U2=24*(26*r**-14-7*r**-8)
    W,H,G=80,48,9            # each panel 5:3, gap between panels
    x0=20; ys=[6,6+H+G,6+2*(H+G)]
    specs=[(U,(-1.3,1.0),[(-1,"−1"),(0,"0"),(1,"1")],"U / ε","a"),
           (U1,(-3.2,3.2),[(-2,"−2"),(0,"0"),(2,"2")],"U′ σ / ε","b"),
           (U2,(-12,72),[(0,"0"),(30,"30"),(60,"60")],"U″ σ² / ε","c")]
    body=[];defs=[];panels=[]
    for i,(y,(v,yr,yt,yl,lab)) in enumerate(zip(ys,specs)):
        p=Panel(x0,y,W,H,(0.95,2.3),yr,f"p{i}");panels.append(p);defs.append(p.clip())
        for rr,c in ((r0,GREEN),(ri,RED)): body.append(line([(p.X(rr),p.y0),(p.X(rr),p.y0+H)],c,0.35,"1.5,1"))
        body.append(p.axes([1,1.5,2],yt,"Separation r / σ",yl,show_x=(i==2)))
        body.append(p.curve(r,v,BLUE,0.8))
        body.append(text(x0-17,y+2,lab,size=6.35,weight="bold"))
    a,b,c=panels
    body.append(text(a.X(r0),a.y0-1.3,"r₀",anchor="middle",fill=GREEN))
    body.append(text(a.X(ri)+1,a.y0-1.3,"inflection",fill=RED))
    # a: bond energy and bond length
    body.append(line([(a.X(r0),a.Y(-1)),(a.X(1.85),a.Y(-1))],GREY,0.3,"1.5,1"))
    body.append(line([(a.X(1.8),a.Y(-0.06)),(a.X(1.8),a.Y(-0.94))],INK,0.4,marker=True))
    body.append(text(a.X(1.85),a.Y(-0.55),"ε: bond energy"))
    body.append(text(a.X(1.3),a.Y(0.55),"minimum at r₀: bond length",fill=GREEN))
    # b: maximum restoring force
    body.append(f'<circle cx="{b.X(ri):.2f}" cy="{b.Y(2.396):.2f}" r="0.9" style="fill:{RED}"/>')
    body.append(text(b.X(1.55),b.Y(2.5),"maximum: largest"))
    body.append(text(b.X(1.55),b.Y(2.5)+5,"restoring force"))
    body.append(text(b.X(1.45),b.Y(-1.4),"U′ = 0 at r₀: no net force",fill=GREEN))
    # c: stiffness and inflection
    body.append(f'<circle cx="{c.X(r0):.2f}" cy="{c.Y(57.15):.2f}" r="0.9" style="fill:{GREEN}"/>')
    body.append(text(c.X(1.3),c.Y(57),"k = U″(r₀) ≈ 57 ε/σ²"))
    body.append(text(c.X(1.3),c.Y(57)+5,"bond stiffness"))
    body.append(f'<circle cx="{c.X(ri):.2f}" cy="{c.Y(0):.2f}" r="0.9" style="fill:{RED}"/>')
    body.append(text(c.X(1.45),c.Y(14),"U″ = 0 at the inflection",fill=RED))
    open(ASSETS/"L26-pair-derivatives.svg","w").write(
        svg(104,ys[2]+H+14,"Lennard–Jones energy, first derivative, and second derivative","".join(body),"".join(defs)))

if __name__=="__main__":
    repulsion_family(); pair_three(); print("saved")
