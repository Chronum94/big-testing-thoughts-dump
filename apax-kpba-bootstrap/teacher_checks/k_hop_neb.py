"""K hop 8c -> neighbouring empty 8c through the 24d window, KFe[Fe(CN)6] (K4 per cell) and
K2-x (K7 per cell, single K vacancy) using a relaxed PET-MAD cell from teacher_checks.py.
Usage: python k_hop_neb.py <model> <relaxed_K1.extxyz> <relaxed_K2.extxyz> <out.json>
Fixed cell, climbing-image NEB, 7 images, fmax 0.05.
"""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE
from petjax import UPETCalculator

model, f1, f2, out = sys.argv[1:5]
ck = f"/home/chronum/scratch/petjax/checkpoints/pet-mad-{model}"
mk = lambda: UPETCalculator.from_checkpoint(ck, stress=False, default_dtype="float64")
res = {}


def run(at0, iK, target_frac, tag, nimg=7):
    a = at0.copy(); a.calc = mk()
    FIRE(a, logfile=None).run(fmax=0.02, steps=800)
    b = at0.copy()
    sp = b.get_scaled_positions(); sp[iK] = target_frac; b.set_scaled_positions(sp)
    b.calc = mk(); FIRE(b, logfile=None).run(fmax=0.02, steps=800)
    images = [a] + [a.copy() for _ in range(nimg)] + [b]
    for im in images[1:-1]:
        im.calc = mk()
    neb = NEB(images, climb=True, allow_shared_calculator=False)
    neb.interpolate(mic=True)
    FIRE(neb, logfile=None).run(fmax=0.05, steps=600)
    E = np.array([im.get_potential_energy() for im in images])
    # K position at the saddle vs 24d
    imax = int(np.argmax(E))
    res[tag] = dict(E_rel=(E - E[0]).tolist(), barrier_eV=float(E.max() - E[0]),
                    dE_end=float(E[-1] - E[0]), saddle_K_frac=images[imax].get_scaled_positions()[iK].tolist())
    print(tag, res[tag], flush=True)
    json.dump(res, open(out, "w"), indent=1)


# K1: K on (1/4,1/4,1/4)+fcc; hop to empty (1/4,1/4,3/4)
at = read(f1)
s = np.array(at.get_chemical_symbols()); sp = at.get_scaled_positions()
iK = [i for i in np.where(s == "K")[0] if np.allclose((sp[i] - 0.25 + 0.5) % 1 - 0.5, 0, atol=0.03)][0]
run(at, iK, sp[iK] + [0, 0, 0.5], "KFe[Fe(CN)6]_K4cell_8c-24d-8c")

# K2 with one K vacancy: remove K at (3/4,3/4,3/4)-type neighbour, hop (1/4,1/4,1/4)->(1/4,1/4,3/4)
at = read(f2)
s = np.array(at.get_chemical_symbols()); sp = at.get_scaled_positions()
Ks = np.where(s == "K")[0]
ivac = [i for i in Ks if np.allclose((sp[i] - [0.25, 0.25, 0.75] + 0.5) % 1 - 0.5, 0, atol=0.03)][0]
target = sp[ivac].copy()
del at[ivac]
s = np.array(at.get_chemical_symbols()); sp = at.get_scaled_positions()
iK = [i for i in np.where(s == "K")[0] if np.allclose((sp[i] - 0.25 + 0.5) % 1 - 0.5, 0, atol=0.03)][0]
run(at, iK, target, "K2Fe[Fe(CN)6]_K7cell_vacancy_hop")
print("DONE")
