"""L26: EAM functions (as drawn in the project bonus notebook) and LJ vs EAM energy curves for SC, BCC, FCC Cu (300 dpi)."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/"assets"

def read_setfl(path):
    lines = Path(path).read_text().splitlines()
    nrho, drho, nr, dr, cutoff = lines[4].split()
    nrho, nr = int(nrho), int(nr); drho, dr, cutoff = float(drho), float(dr), float(cutoff)
    values = np.array(" ".join(lines[6:]).split(), dtype=float)
    r_grid = np.arange(nr) * dr; rho_grid = np.arange(nrho) * drho
    embedding = values[:nrho]; density = values[nrho:nrho + nr]; r_phi = values[nrho + nr:nrho + 2 * nr]
    pair = np.empty(nr); pair[1:] = r_phi[1:] / r_grid[1:]; pair[0] = pair[1]
    return dict(phi=pair, density=density, embed=embedding, r=r_grid, rho=rho_grid, cutoff=cutoff)
eam = read_setfl(ROOT/"activities/Cu_mishin1.eam.alloy")

# 1. EAM functions, same drawing as the project bonus notebook (reference_curves cell)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
m = (eam["r"] >= 1.8) & (eam["r"] <= eam["cutoff"])
ax[0].plot(eam["r"][m], eam["phi"][m], color="tab:red"); ax[0].set(xlabel="Pair distance r (Å)", ylabel="Pair energy φ(r) (eV)")
ax[1].plot(eam["r"][m], eam["density"][m], color="tab:blue"); ax[1].set(xlabel="Neighbour distance r (Å)", ylabel="Density contribution ρ(r)")
ax[2].plot(eam["rho"], eam["embed"], color="tab:green"); ax[2].set(xlabel="Total local density ρ", ylabel="Embedding energy F(ρ) (eV)")
ax[0].set_title("Pair energy φ(rᵢⱼ)"); ax[1].set_title("Neighbour density ρ(r)"); ax[2].set_title("Embedding energy F(ρ)")
lim = ax[2].get_xlim(); rmin = float(eam["rho"][eam["embed"].argmin()])
ax[2].axvspan(lim[0], rmin, color="#c5e1a5", alpha=0.5, zorder=0); ax[2].axvspan(rmin, lim[1], color="#ffccbc", alpha=0.5, zorder=0); ax[2].set_xlim(lim)
ax[2].text((lim[0]+rmin)/2, 0.97, "More neighbours\n→ more stable", transform=ax[2].get_xaxis_transform(), fontsize=8, ha="center", va="top")
ax[2].text((rmin+lim[1])/2, 0.97, "Too crowded\n→ less stable", transform=ax[2].get_xaxis_transform(), fontsize=8, ha="center", va="top")
fig.tight_layout(); fig.savefig(A/"L26-eam-functions.png", dpi=300); plt.close(fig)

# 2. LJ vs EAM for SC, BCC, FCC
def neighbour_distances(structure, d, rc):
    if structure == "FCC":
        a = d*np.sqrt(2); cell = a*np.eye(3); basis = a*np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    elif structure == "BCC":
        a = d*2/np.sqrt(3); cell = a*np.eye(3); basis = a*np.array([[0,0,0],[.5,.5,.5]])
    else:
        cell = d*np.eye(3); basis = np.zeros((1,3))
    n = int(np.ceil(rc/d)) + 2
    shifts = np.array(np.meshgrid(*[np.arange(-n,n+1)]*3)).reshape(3,-1).T @ cell
    r = np.linalg.norm((basis[None]+shifts[:,None]).reshape(-1,3) - basis[0], axis=1)
    return r[(r > 1e-8) & (r < rc)]
eps, sig = 0.44101445113186266, 2.3375899488677705   # LJ fitted to fcc Cu: a0 = 3.615 Å, Ecoh = 3.54 eV
def E(model, s, d):
    if model == "LJ":
        r = neighbour_distances(s, d, 3*sig); q = (sig/r)**6; x = r*r; ro2, rc2 = (2*sig)**2, (3*sig)**2
        sw = np.where(x <= ro2, 1., (rc2-x)**2*(rc2+2*x-3*ro2)/(rc2-ro2)**3)
        return .5*np.sum(4*eps*(q*q-q)*sw)
    r = neighbour_distances(s, d, eam["cutoff"])
    rho = np.sum(np.interp(r, eam["r"], eam["density"]))
    return .5*np.sum(np.interp(r, eam["r"], eam["phi"])) + np.interp(rho, eam["rho"], eam["embed"])
crys = Image.open(A/"L25-crystal-structures.png")
crops = {"SC":(80,80,575,525), "BCC":(45,630,550,1085), "FCC":(45,1125,580,1605)}
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
d = np.linspace(2.1, 3.3, 241)
for a, s in zip(axes, ["SC","BCC","FCC"]):
    for model, c, ls in (("LJ","#b5473a","--"),("EAM","#1f77b4","-")):
        y = np.array([E(model, s, x) for x in d]); a.plot(d, y, c=c, ls=ls, lw=2.2, label=model)
        k = y.argmin(); a.plot(d[k], y[k], "o", c=c, ms=5)
        print(s, model, f"d_min={d[k]:.3f} Å, E_min={y[k]:.3f} eV/atom")
    a.set_title(s, fontsize=14); a.set_xlabel("nearest-neighbour distance (Å)"); a.set_xlim(2.1, 3.3); a.set_ylim(-3.8, -1.0)
    a.spines[["top","right"]].set_visible(False); a.grid(alpha=.2)
    ins = a.inset_axes([0.52, 0.52, 0.45, 0.45]); ins.imshow(crys.crop(crops[s])); ins.axis("off")
axes[0].set_ylabel("energy per atom (eV)"); axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(A/"L26-lj-vs-eam-phases.png", dpi=300)
