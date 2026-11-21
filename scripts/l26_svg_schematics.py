"""L26 schematics in the course SVG style (mm units, Arial, thin ink lines, gradient atoms):
force-field terms, stress tensor with C12/C44, and two MLIP designs."""
from pathlib import Path
import numpy as np
from l26_svg_figures import text, line, svg, INK, GREY, LGREY, RED, BLUE, FS
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/"assets"
SM=3.4   # small label size
GRAD=('<radialGradient id="ab" cx="0.5" cy="0.5" r="0.5" fx="0.35" fy="0.32"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#8a8f98"/></radialGradient>'
      '<radialGradient id="ar" cx="0.5" cy="0.5" r="0.5" fx="0.35" fy="0.32"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#b5473a"/></radialGradient>'
      '<radialGradient id="an" cx="0.5" cy="0.5" r="0.5" fx="0.35" fy="0.32"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#1f77b4"/></radialGradient>')
def atom(x,y,r=2.2,g="ab"): return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" style="fill:url(#{g});stroke:{INK};stroke-width:0.35"/>'
import re
def M(s,size=3.0):
    """mini markup: a_{b} subscript, a^{b} superscript; the rest of the string is reset to the baseline"""
    out=""; pos=0; shift=0.0
    for m in re.finditer(r"([_^])\{([^}]*)\}",s):
        plain=s[pos:m.start()]
        if plain: out+=f'<tspan dy="{-shift:g}">{plain}</tspan>' if shift else plain; shift=0.0 if plain else shift
        d=1.0 if m.group(1)=="_" else -1.6
        out+=f'<tspan style="font-size:{size}px" dy="{d-shift:g}">{m.group(2)}</tspan>'; shift=d
        pos=m.end()
    rest=s[pos:]
    if rest: out+=f'<tspan dy="{-shift:g}">{rest}</tspan>' if shift else rest
    return out
def arrow(p,q,c=INK,w=0.5): return line([p,q],c,w,marker=True)
def dbl(p,q,c=INK,w=0.4): return line([p,q],c,w,marker=True).replace("<path ",'<path marker-start="url(#tri)" ')
def rect(x,y,w,h,fill="none",stroke=INK,sw=0.35,rx=0,dash=None):
    d=f";stroke-dasharray:{dash}" if dash else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" style="fill:{fill};stroke:{stroke};stroke-width:{sw}{d}"/>'
def mini(x0,y0,w,h,xs,ys,xl,zero=False):
    """small energy plot: axes, curve, axis letters"""
    xs=np.asarray(xs); ys=np.asarray(ys)
    lo,hi=np.nanmin(ys),np.nanmax(ys); ys=np.clip(ys,lo,hi)
    X=x0+(xs-xs.min())/(xs.max()-xs.min())*w; Y=y0+h-(ys-lo)/(hi-lo)*h
    out=[line([(x0,y0-1),(x0,y0+h),(x0+w+1,y0+h)],INK,0.4)]
    if zero and lo<0<hi:
        yz=y0+h-(0-lo)/(hi-lo)*h; out.append(line([(x0,yz),(x0+w,yz)],LGREY,0.3,"1.2,0.8"))
    out.append(line(list(zip(X,Y)),BLUE,0.7))
    out.append(text(x0+w/2,y0+h+4.2,xl,size=SM,anchor="middle",style="italic"))
    out.append(text(x0-1.6,y0+h/2+1.2,"U",size=SM,anchor="end",style="italic"))
    return "".join(out)

def forcefield():
    W,H=186,98; cw,ch=62,48; b=[]
    for k in range(6):
        cx,cy=(k%3)*cw,(k//3)*ch
        sx,sy=cx+15,cy+20          # sketch centre
        px,py,pw,ph=cx+33,cy+9,24,20   # mini plot
        if k==0:
            name="bond stretching"; f=M("U = ½ k_{b}(b − b_{0})^{2}")
            t=np.linspace(0,1,120); xx=sx-7+14*t; yy=sy+0.9*np.sin(2*np.pi*6*t)*((t>.15)&(t<.85))
            b.append(line(list(zip(xx,yy)),GREY,0.35)); b+= [atom(sx-7,sy),atom(sx+7,sy)]
            b.append(dbl((sx-7,sy+5),(sx+7,sy+5))); b.append(text(sx,sy+9.5,"b",size=SM,anchor="middle",style="italic"))
            x=np.linspace(-1,1,100); b.append(mini(px,py,pw,ph,x,x**2,"b"))
        elif k==1:
            name="angle bending"; f=M("U = ½ k_{θ}(θ − θ_{0})^{2}")
            c=(sx,sy+4); p=(sx-8,sy-4); q=(sx+8,sy-4)
            b+=[line([c,p],GREY,0.8),line([c,q],GREY,0.8),atom(*p),atom(*q),atom(*c)]
            b.append(f'<path d="M{sx-3.5:.2f},{sy+0.5:.2f} A5,5 0 0,1 {sx+3.5:.2f},{sy+0.5:.2f}" style="fill:none;stroke:{RED};stroke-width:0.45"/>')
            b.append(text(sx,sy-1.5,"θ",size=SM,anchor="middle",fill=RED,style="italic"))
            x=np.linspace(-1,1,100); b.append(mini(px,py,pw,ph,x,x**2,"θ"))
        elif k==2:
            name="dihedral torsion"; f=M("U = k_{φ}[1 + cos(nφ − δ)]")
            P=[(sx-12,sy-5),(sx-5,sy+1),(sx+2,sy+1),(sx+9,sy+7)]
            for i in range(3): b.append(line([P[i],P[i+1]],GREY,0.8))
            b.append(line([P[2],(sx+9,sy-5)],GREY,0.35,"1,0.7"))
            b+=[atom(*p) for p in P]
            b.append(f'<path d="M{sx+6:.2f},{sy-2.5:.2f} A5,5 0 0,1 {sx+6:.2f},{sy+4.5:.2f}" style="fill:none;stroke:{RED};stroke-width:0.45"/>')
            b.append(text(sx+9,sy+2.5,"φ",size=SM,fill=RED,style="italic"))
            x=np.linspace(0,2*np.pi,200); b.append(mini(px,py,pw,ph,x,1+np.cos(3*x),"φ"))
        elif k==3:
            name="improper torsion"; f=M("U = ½ k_{ξ}(ξ − ξ_{0})^{2}")
            L=[(sx-9,sy+6),(sx+9,sy+6),(sx,sy-8)]; c=(sx,sy+1)
            b.append(line([L[0],L[1],L[2],L[0]],LGREY,0.3,"1,0.7"))
            for p in L: b.append(line([c,p],GREY,0.8))
            b+=[atom(*p) for p in L]+[atom(*c)]
            b.append(arrow((sx+3.5,sy+1),(sx+3.5,sy-3.5),RED,0.4)); b.append(text(sx+5,sy-1,"ξ",size=SM,fill=RED,style="italic"))
            x=np.linspace(-1,1,100); b.append(mini(px,py,pw,ph,x,x**2,"ξ"))
        elif k==4:
            name="electrostatic (Coulomb)"; f=M("U = q_{i}q_{j} / (4πε_{0}r)")
            b+=[atom(sx-7,sy,g="ar"),atom(sx+7,sy,g="an")]
            b.append(text(sx-7,sy+1.2,"+",size=SM,anchor="middle")); b.append(text(sx+7,sy+1.2,"−",size=SM,anchor="middle"))
            b.append(dbl((sx-7,sy+5),(sx+7,sy+5))); b.append(text(sx,sy+9.5,"r",size=SM,anchor="middle",style="italic"))
            r=np.linspace(0.35,5,200); b.append(mini(px,py,pw,ph,r,-1/r,"r",zero=True))
        else:
            name="Lennard–Jones"; f=M("U = 4ε[(σ/r)^{12} − (σ/r)^{6}]")
            b+=[atom(sx-7,sy),atom(sx+7,sy)]
            b.append(dbl((sx-7,sy+5),(sx+7,sy+5))); b.append(text(sx,sy+9.5,"r",size=SM,anchor="middle",style="italic"))
            r=np.linspace(0.97,2.6,200); u=np.clip(4*(r**-12-r**-6),-1.1,1.0); b.append(mini(px,py,pw,ph,r,u,"r",zero=True))
        b.append(text(cx+3,cy+5,name,fill=INK))
        b.append(text(cx+cw/2,cy+40,f,size=3.8,anchor="middle"))
    b.append(line([(0,ch-2),(W,ch-2)],LGREY,0.25,"1,1"))
    open(A/"L26-forcefield-terms.svg","w").write(svg(W,H,"Energy terms of a molecular force field","".join(b),GRAD))

def stress():
    W,H=190,78; b=[]
    # oblique cube: x right, y back (up-right), z up
    def P(x,y,z): return np.array([22+30*x+13*y, 60-30*z-9*y])
    F={"x":[(1,0,0),(1,1,0),(1,1,1),(1,0,1)],"y":[(0,0,0),(1,0,0),(1,0,1),(0,0,1)],"z":[(0,0,1),(1,0,1),(1,1,1),(0,1,1)]}
    # visible faces: front (y=0 plane, normal -y), right (x=1), top (z=1)
    fills={"y":"#f2f4f6","x":"#e3e7eb","z":"#f8f9fa"}
    for k in ("y","x","z"):
        pts=[P(*v) for v in F[k]]
        b.append('<path d="M'+" L".join(f"{p[0]:.2f},{p[1]:.2f}" for p in pts)+f' Z" style="fill:{fills[k]};stroke:{INK};stroke-width:0.4"/>')
    U={"x":np.array([1,0]),"y":(P(0,1,0)-P(0,0,0))/np.linalg.norm(P(0,1,0)-P(0,0,0)),"z":np.array([0,-1])}
    cen={"x":P(1,.5,.5),"z":P(.5,.5,1),"y":P(.5,0,.5)}
    lab={("x","x"):(1.5,1.3),("x","y"):(1.5,2.2),("x","z"):(-8.5,-0.5),("z","z"):(1.5,0),("z","x"):(1,1.5),("z","y"):(-1,-2.2),
         ("y","y"):(-6,2),("y","x"):(1,1.5),("y","z"):(1.5,0)}
    for f in ("x","z","y"):
        c=cen[f]; b.append(f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="0.5" style="fill:{INK}"/>')
        for d in "xyz":
            n=(-U["y"] if (f=="y" and d=="y") else U[d]) if f=="y" else U[d]
            if f=="y" and d!="y": n=U[d]
            Lg=10 if d==f else 7
            q=c+Lg*n; col=RED if d==f else BLUE
            b.append(arrow(tuple(c),tuple(q),col,0.45))
            o=lab[(f,d)]; b.append(text(q[0]+o[0],q[1]+o[1],M("σ_{"+f+d+"}"),size=3.6,fill=col,style="italic"))
    o=np.array([4,74])
    for d,v in (("x",U["x"]),("y",U["y"]),("z",U["z"])):
        q=o+6*v; b.append(arrow(tuple(o),tuple(q),INK,0.35)); b.append(text(q[0]+0.8*v[0]+1,q[1]+0.8*v[1]+1,d,size=SM,style="italic"))
    b.append(text(30,80,"red: normal stress",size=SM,anchor="middle",fill=RED)); b.append(text(66,80,"blue: shear stress",size=SM,anchor="middle",fill=BLUE))
    # stretch panel
    x0,y0,s=96,22,26
    b.append(rect(x0,y0,s,s,stroke=LGREY,sw=0.35,dash="1.2,0.8"))
    b.append(rect(x0,y0,s*1.3,s,fill="#e3e7eb",stroke=INK,sw=0.4))
    for xx in (x0+8,x0+26): b+= [arrow((xx,y0-0.5),(xx,y0-7),RED,0.45),arrow((xx,y0+s+0.5),(xx,y0+s+7),RED,0.45)]
    b+= [arrow((x0+s*1.3+0.5,y0+s/2),(x0+s*1.3+7,y0+s/2),GREY,0.45),arrow((x0-0.5,y0+s/2),(x0-7,y0+s/2),GREY,0.45)]
    b.append(text(x0+s*1.3+2,y0+s/2-2,M("ε_{xx}"),size=3.6,fill=GREY,style="italic"))
    b.append(text(x0+28,y0-4,M("σ_{yy}"),size=3.6,fill=RED,style="italic"))
    b.append(text(x0+s*0.65,y0+s+15,M("σ_{yy} = C_{12} ε_{xx}"),anchor="middle"))
    b.append(text(x0+s*0.65,y0-12,"stretch along x",anchor="middle"))
    # shear panel
    x0=150
    b.append(rect(x0,y0,s,s,stroke=LGREY,sw=0.35,dash="1.2,0.8"))
    pts=[(x0,y0+s),(x0+s,y0+s),(x0+s+8,y0),(x0+8,y0)]
    b.append('<path d="M'+" L".join(f"{p[0]:.2f},{p[1]:.2f}" for p in pts)+f' Z" style="fill:#e3e7eb;stroke:{INK};stroke-width:0.4"/>')
    b.append(arrow((x0+10,y0-3),(x0+30,y0-3),BLUE,0.45)); b.append(arrow((x0+22,y0+s+3),(x0+2,y0+s+3),BLUE,0.45))
    b.append(text(x0+31.5,y0-1.8,M("σ_{xy}"),size=3.6,fill=BLUE,style="italic"))
    b.append(text(x0-6,y0+6,M("γ_{xy}"),size=3.6,fill=GREY,style="italic"))
    b.append(text(x0+s/2+4,y0+s+15,M("σ_{xy} = C_{44} γ_{xy}"),anchor="middle"))
    b.append(text(x0+s/2+4,y0-12,"shear in the x–y plane",anchor="middle"))
    open(A/"L26-stress-tensor.svg","w").write(svg(W,H+6,"Stress components and the strains behind C12 and C44","".join(b),GRAD))

def mlip():
    W,H=190,66; b=[]
    def box(x,y,w,h,t,fill="#f2f4f6",size=3.6):
        b.append(rect(x-w/2,y-h/2,w,h,fill=fill,sw=0.35,rx=1.2))
        lines=t.split("\n")
        for i,s in enumerate(lines): b.append(text(x,y+1.3+(i-(len(lines)-1)/2)*4.2,s,size=size,anchor="middle"))
    b.append(text(48,6,"atom-centred neural network",anchor="middle"))
    ys=[18,32,46]
    for i,y in enumerate(ys):
        b.append(atom(6,y,2.6,"ar" if i==0 else "ab")); b.append(text(6,y+1.2,str(i+1),size=SM,anchor="middle"))
        b.append(arrow((9,y),(13.5,y),INK,0.4)); box(24,y,21,9,"neighbour\ndescriptor",size=3.1)
        b.append(arrow((34.5,y),(39,y),INK,0.4))
        xs=[42,48,54]; ns=[3,4,1]
        nodes=[[(xx,y+(k-(n-1)/2)*2.4) for k in range(n)] for xx,n in zip(xs,ns)]
        for L1,L2 in zip(nodes[:-1],nodes[1:]):
            for p in L1:
                for q in L2: b.append(line([p,q],LGREY,0.2))
        for L in nodes:
            for p in L: b.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="0.7" style="fill:white;stroke:{INK};stroke-width:0.3"/>')
        b.append(arrow((55.5,y),(60,y),INK,0.4)); b.append(text(64,y+1.3,M("E_{"+str(i+1)+"}"),style="italic"))
        b.append(arrow((70,y),(80,32+(y-32)*0.25),INK,0.4))
    b.append(f'<circle cx="84" cy="32" r="3.6" style="fill:white;stroke:{INK};stroke-width:0.35"/>'); b.append(text(84,33.6,"Σ",anchor="middle"))
    b.append(text(84,41,"total E",size=SM,anchor="middle"))
    b.append(text(45,60,"same small network for every atom of one element",size=SM,anchor="middle",fill=GREY))
    # message passing
    b.append(line([(97,10),(97,62)],LGREY,0.25,"1,1"))
    b.append(text(145,6,"message-passing graph network (MACE)",anchor="middle"))
    c=np.array([118,32])
    for t in np.linspace(0,2*np.pi,7)[:-1]+0.3:
        p=c+13*np.array([np.cos(t),np.sin(t)])
        b.append(line([tuple(c),tuple(p)],LGREY,0.35))
        u=(c-p)/np.linalg.norm(c-p); b.append(arrow(tuple(p+3*u),tuple(c-4*u),RED,0.4))
        b.append(atom(*p,2.4))
    b.append(atom(*c,3.0,"ar")); b.append(text(c[0],c[1]+1.3,"i",size=SM,anchor="middle",style="italic"))
    b.append(text(120,54,"neighbours send messages",size=SM,anchor="middle",fill=GREY))
    b.append(text(120,58,"(distances and angles)",size=SM,anchor="middle",fill=GREY))
    box(155,18,31,8,"update atom i"); box(155,32,31,8,"repeat the update",fill="white"); box(155,46,31,8,M("atom energy E_{i}"),fill="white")
    b.append(arrow((133,24),(139,19.5),INK,0.4)); b.append(arrow((155,22.3),(155,27.5),INK,0.4)); b.append(arrow((155,36.3),(155,41.5),INK,0.4))
    b.append(arrow((171,46),(176,46),INK,0.4))
    b.append(f'<circle cx="180" cy="46" r="3.6" style="fill:white;stroke:{INK};stroke-width:0.35"/>'); b.append(text(180,47.6,"Σ",anchor="middle"))
    b.append(text(180,55,"total E",size=SM,anchor="middle"))
    open(A/"L26-mlip-types.svg","w").write(svg(W,H,"Two designs of machine-learned interatomic potentials","".join(b),GRAD))
if __name__=="__main__":
    forcefield(); stress(); mlip(); print("saved")
