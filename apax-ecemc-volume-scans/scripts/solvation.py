"""Li+ first-shell composition: carbonyl vs ether O per solvent (EC/EMC/DMC) and F per anion (PF6/LiF/HF).

usage: python solvation.py [--rc 2.6] TRAJ.extxyz [TRAJ ...]    (frames need arrays['mol_id'])
O type: within its own molecule, O bonded (< 1.6 A) to one C = carbonyl, to two C = ether. Only free Li+ ions (single-atom
molecules) are analysed; the Li of LiF is skipped. Prints mean coordination number per site type and the fraction of
Li-O contacts that are carbonyl, per solvent and overall.
"""
import argparse
import collections

import numpy as np
from ase.io import read

ap = argparse.ArgumentParser(); ap.add_argument("traj", nargs="+"); ap.add_argument("--rc", type=float, default=2.6)
a_ = ap.parse_args()
NAMES = {"C3H4O3": "EC", "C4H8O3": "EMC", "C3H6O3": "DMC", "F6P1": "PF6", "Li1": "Li", "F1Li1": "LiF", "F1H1": "HF"}


def site_types(a):
    mol, sym = a.arrays["mol_id"], np.array(a.get_chemical_symbols()); t = np.full(len(a), "", dtype=object)
    d = a.get_all_distances(mic=True)
    for g in np.unique(mol):
        idx = np.where(mol == g)[0]; c = collections.Counter(sym[idx])
        name = NAMES.get("".join(f"{e}{c[e]}" for e in sorted(c)), "?")
        for i in idx:
            if sym[i] == "O":
                nC = sum(1 for j in idx if sym[j] == "C" and d[i, j] < 1.6)
                t[i] = f"{name} O={'C' if nC == 1 else 'ether'}" if nC else f"{name} O?"
            elif sym[i] == "F":
                t[i] = f"{name} F"
            elif sym[i] == "Li" and name == "Li":
                t[i] = "Li+"
    return t


cn, nli = collections.Counter(), 0
for f in a_.traj:
    for a in read(f, ":"):
        t = site_types(a); li = np.where(t == "Li+")[0]
        if not len(li):
            continue
        d = a.get_all_distances(mic=True)
        for i in li:
            nli += 1
            for j in np.where((d[i] < a_.rc) & (np.arange(len(a)) != i))[0]:
                if t[j]:
                    cn[t[j]] += 1
print(f"{nli} Li+ samples, r_c {a_.rc} A")
print("| site | CN per Li+ |\n|---|---|")
for k, v in sorted(cn.items(), key=lambda x: -x[1]):
    print(f"| {k} | {v / max(nli, 1):.3f} |")
for s in ("EC", "EMC", "DMC", "all"):
    c = sum(v for k, v in cn.items() if k.endswith("O=C") and (s == "all" or k.startswith(s + " ")))
    e = sum(v for k, v in cn.items() if k.endswith("O=ether") and (s == "all" or k.startswith(s + " ")))
    if c + e:
        print(f"{s}: carbonyl fraction of Li-O contacts {c / (c + e):.3f} (carbonyl {c / nli:.3f}, ether {e / nli:.3f} per Li+)")
