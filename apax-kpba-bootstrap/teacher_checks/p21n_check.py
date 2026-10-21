"""Does the teacher keep the K-ion slide (P2_1/n) phase of K2Fe[Fe(CN)6] and K2Mn[Fe(CN)6]?
Builds the 32-atom P2_1/n cell from Cattermull et al. JACS 145, 24249 (2023) SI Tables S6/S7
(doi 10.1021/jacs.3c08751), relaxes positions+cell (FrechetCellFilter+FIRE, fmax 0.02), reports
E/atom vs the relaxed cubic Fm-3m K2Fe cell (8c K) from teacher_checks.py, K off-centre distance,
bond lengths and cell. Also relaxes the same cell with K moved to the ideal (3/4,1/2,0) centre.
Usage: python p21n_check.py <model> <cubic_relaxed_K2.extxyz> <out.json>
"""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from ase import Atoms
from ase.io import read
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from petjax import UPETCalculator

model, cubic, out = sys.argv[1:4]
calc = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/pet-mad-{model}", stress=True, default_dtype="float64")

OPS = [lambda p: p, lambda p: np.array([-p[0] + .5, p[1] + .5, -p[2] + .5]),
       lambda p: -p, lambda p: np.array([p[0] + .5, -p[1] + .5, p[2] + .5])]
DATA = {  # CAT23 SI Table S6 (K2Fe) and S7 (K2Mn); M2 is the N-bound metal
 "K2Fe[Fe(CN)6]": dict(M2="Fe", cell=(10.063, 7.276, 6.948, 89.7977),
   sites=[("Fe", (0, 0, 0)), ("Fe", (.5, 0, 0)), ("K", (.7499, .56149, -.0197)),
          ("C", (-.0322, .8580, .7744)), ("C", (-.0318, .2110, .8542)), ("C", (.8123, -.0004, .0619)),
          ("N", (-.0437, .7540, .6584)), ("N", (.9485, .3353, .7609)), ("N", (.7045, -.0312, .0913))]),
 "K2Mn[Fe(CN)6]": dict(M2="Mn", cell=(10.0991, 7.3349, 6.948, 90.0118),
   sites=[("Fe", (0, 0, 0)), ("Mn", (.5, 0, 0)), ("K", (.7517, .56517, -.0204)),
          ("C", (-.0344, .8550, .7713)), ("C", (-.0356, .2136, .8554)), ("C", (.8174, -.0133, .0667)),
          ("N", (-.0517, .7642, .6491)), ("N", (.9429, .3412, .7640)), ("N", (.7070, -.0294, .1032))]),
}


def build(d):
    a, b, c, beta = d["cell"]
    br = np.radians(beta)
    cellm = np.array([[a, 0, 0], [0, b, 0], [c * np.cos(br), 0, c * np.sin(br)]])
    sym, fr = [], []
    for el, p in d["sites"]:
        for op in OPS:
            q = np.mod(op(np.array(p, float)), 1.0)
            if not any(np.allclose((q - f + .5) % 1 - .5, 0, atol=1e-3) for s_, f in zip(sym, fr) if s_ == el):
                sym.append(el); fr.append(q)
    return Atoms(sym, scaled_positions=fr, cell=cellm, pbc=True)


def bonds(at, M2):
    i, j, d = neighbor_list("ijd", at, 2.6); s = at.get_chemical_symbols()
    r = {}
    for a_, b_, dd in zip(i, j, d):
        p = s[a_] + "-" + s[b_]
        if p in ("Fe-C", "C-N", M2 + "-N"):
            r.setdefault(p, []).append(dd)
    return {k: [float(np.min(v)), float(np.max(v))] for k, v in r.items()}


def koff(at):
    sp = at.get_scaled_positions()[np.array(at.get_chemical_symbols()) == "K"]
    # cavity centre = mean of the 8 surrounding M positions ~ nearest (x in {1/4,3/4}, y in {0,1/2}, z in {0,1/2})
    cands = np.array([[x, y, z] for x in (.25, .75) for y in (0, .5) for z in (0, .5)])
    ds = []
    for p in sp:
        dd = [np.linalg.norm((((p - c_ + .5) % 1) - .5) @ at.cell.array) for c_ in cands]
        ds.append(min(dd))
    return [float(np.min(ds)), float(np.max(ds))]


res = {}
cub = read(cubic); cub.calc = calc
res["cubic_ref_E_per_atom"] = float(cub.get_potential_energy() / len(cub))
for name, d in DATA.items():
    for variant in ("exp_slid", "K_centred"):
        at = build(d)
        if variant == "K_centred":
            sp = at.get_scaled_positions(); m = np.array(at.get_chemical_symbols()) == "K"
            sp[m] = np.round(sp[m] * 4) / 4  # snap to cavity centre (x in 1/4,3/4; y,z in 0,1/2)
            at.set_scaled_positions(sp)
        r0 = dict(n=len(at), formula=at.get_chemical_formula(), koff_start=koff(at), bonds_start=bonds(at, d["M2"]))
        at.calc = calc
        opt = FIRE(FrechetCellFilter(at), logfile=None); conv = opt.run(fmax=0.02, steps=2000)
        r0.update(converged=bool(conv), steps=opt.nsteps, E_per_atom=float(at.get_potential_energy() / len(at)),
                  cellpar=at.cell.cellpar().tolist(), pseudocubic_a=float((2 * at.get_volume()) ** (1 / 3)),
                  koff_end=koff(at), bonds_end=bonds(at, d["M2"]))
        at.write(f"/home/chronum/scratch/pba/lit/relaxed_{model}_p21n_{name[:2]}{d['M2']}_{variant}.extxyz")
        res[f"{name}_{variant}"] = r0
        print(name, variant, json.dumps(r0), flush=True)
        json.dump(res, open(out, "w"), indent=1)
print("DONE")
