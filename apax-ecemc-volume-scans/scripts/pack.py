"""Pack small periodic LiPF6 / EC:EMC 3:7 boxes with packmol (pbc, tolerance 2.0 A) -> pack/<comp>_rho<r>_s<seed>.extxyz
Compositions by molecule count: elec = 1 LiPF6 + 3 EC + 7 EMC (143 atoms), solv = 3 EC + 7 EMC (135), conc = 2 LiPF6 + 3 EC + 7 EMC (151).
Each atom carries arrays['mol_id'] (molecule index; Li+ and PF6- are their own molecules) for rigid-molecule volume scans.
Variants ec4/dmc/lif/hf swap or add one molecule so the per-element energy regression has full rank (check: --rank).
usage: python pack.py [--rank]   (densities 1.10/1.20/1.30 g/cm3, seeds 1-2)"""
import itertools
import os
import subprocess

import numpy as np
from ase.io import read

PACKMOL = "/home/chronum/miniconda3/envs/packmol/bin/packmol"
MOLS = {k: read(f"mol/{k}.xyz") for k in ("EC", "EMC", "DMC", "PF6", "Li", "LiF", "HF")}
COMPS = {"elec": {"Li": 1, "PF6": 1, "EC": 3, "EMC": 7}, "solv": {"EC": 3, "EMC": 7}, "conc": {"Li": 2, "PF6": 2, "EC": 3, "EMC": 7},
         "ec4": {"Li": 1, "PF6": 1, "EC": 4, "EMC": 6}, "dmc": {"Li": 1, "PF6": 1, "EC": 3, "EMC": 6, "DMC": 1},
         "lif": {"Li": 1, "PF6": 1, "LiF": 1, "EC": 3, "EMC": 7}, "hf": {"Li": 1, "PF6": 1, "HF": 1, "EC": 3, "EMC": 7}}
import sys as _s
if "--rank" in _s.argv:
    E = ["Li", "P", "F", "C", "H", "O"]
    M = np.array([[sum(n * MOLS[k].get_chemical_symbols().count(e) for k, n in c.items()) for e in E] for c in COMPS.values()])
    print(M); print("rank", np.linalg.matrix_rank(M), "of", len(E)); _s.exit()
os.makedirs("pack/tmp", exist_ok=True)
for (comp, cnt), rho, seed in itertools.product(COMPS.items(), (1.10, 1.20, 1.30), (1, 2)):
    mass = sum(n * MOLS[k].get_masses().sum() for k, n in cnt.items())
    L = (mass / 0.602214 / rho) ** (1 / 3)
    tag = f"{comp}_rho{rho:.2f}_s{seed}"; out = f"pack/tmp/{tag}.xyz"
    inp = f"tolerance 2.0\nfiletype xyz\noutput {out}\nseed {seed}\npbc 0. 0. 0. {L:.4f} {L:.4f} {L:.4f}\n"
    for k, n in cnt.items():
        inp += f"structure mol/{k}.xyz\n  number {n}\nend structure\n"
    open(f"pack/tmp/{tag}.inp", "w").write(inp)
    r = subprocess.run([PACKMOL], stdin=open(f"pack/tmp/{tag}.inp"), capture_output=True, text=True)  # packmol seeks in stdin: needs a file
    if "Success!" not in r.stdout:
        print(tag, "FAILED"); continue
    a = read(out); a.set_cell([L, L, L]); a.pbc = True; a.wrap()
    mol = np.concatenate([np.full(len(MOLS[k]), i) for i, k in enumerate(k for k, n in cnt.items() for _ in range(n))])
    a.arrays["mol_id"] = mol; a.info.update(comp=comp, rho=rho, seed=seed)
    d = a.get_all_distances(mic=True); np.fill_diagonal(d, 9); inter = d[mol[:, None] != mol[None, :]].min()
    a.write(f"pack/{tag}.extxyz"); print(f"{tag}: {len(a)} atoms, L {L:.2f} A, min intermolecular d {inter:.2f} A")
