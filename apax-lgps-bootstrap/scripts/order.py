"""Enumerate ordered 50-atom LGPS cells: Ge/P on the 4d sites x random Li orderings (20 of 32 sites,
no Li-Li pair below DMIN), screen by pet-mad-s single points.

usage: python order.py N_LI_SAMPLES out.extxyz
"""
import itertools
import sys

import numpy as np
from ase.io import write
from petjax import UPETCalculator

sys.path.insert(0, "/home/chronum/scratch/lgps")
from sites import expand  # noqa: E402

DMIN, N_LI = 1.8, 20
rng = np.random.default_rng(0)
full, lab, occ = expand()
lab = lab.astype(str)
d = full.get_all_distances(mic=True)
li = np.flatnonzero(np.char.startswith(lab, "Li"))
gep = np.flatnonzero(lab == "GeP")
rest = np.flatnonzero(~np.char.startswith(lab, "Li") & (lab != "GeP"))


def sample_li():
    """Occupancy-weighted greedy pick of N_LI Li sites with no close pair; None if it gets stuck."""
    chosen, cand = [], list(li)
    while len(chosen) < N_LI:
        ok = [s for s in cand if all(d[s, c] >= DMIN for c in chosen)]
        if not ok:
            return None
        w = occ[ok] / occ[ok].sum()
        chosen.append(ok[rng.choice(len(ok), p=w)]); cand.remove(chosen[-1])
    return tuple(sorted(chosen))


li_sets = set()
for _ in range(50 * int(sys.argv[1])):
    s = sample_li()
    if s: li_sets.add(s)
    if len(li_sets) >= int(sys.argv[1]): break
print(f"{len(li_sets)} distinct Li orderings", flush=True)

calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-s")
out = []
for ge in itertools.combinations(range(4), 2):
    for k, s in enumerate(sorted(li_sets)):
        a = full[list(rest) + list(gep) + list(s)]
        sym = a.get_chemical_symbols()
        for g in range(4):
            sym[len(rest) + g] = "Ge" if g in ge else "P"
        a.set_chemical_symbols(sym)
        a.calc = calc
        e = a.get_potential_energy()
        a.info.update(ge=str(ge), li_id=k, li_sites=",".join(map(str, s)), e_s=e)
        a.calc = None
        out.append(a)
    es = [x.info["e_s"] for x in out if x.info["ge"] == str(ge)]
    print(f"Ge on 4d {ge}: min {min(es):.3f} median {np.median(es):.3f} eV", flush=True)
out.sort(key=lambda x: x.info["e_s"])
write(sys.argv[2], out)
print("lowest:", [(x.info["ge"], x.info["li_id"], round(x.info["e_s"] - out[0].info["e_s"], 3)) for x in out[:10]])
print(f"{out[0].get_chemical_formula()} spread {out[-1].info['e_s'] - out[0].info['e_s']:.2f} eV")
