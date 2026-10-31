"""Full 6x6 elastic tensor for structures read from files (any crystal class), relaxed- and clamped-ion.

usage: elastic_gen.py teacher|apax PATH OUT.jsonl NAME=FILE@INDEX [NAME=FILE@INDEX ...]
Full cell + ion relaxation first (FrechetCellFilter, no symmetry imposed), then the same strain set as elastic.py.
Derived: Voigt B and G, min eigenvalue of sym(C) (Born stability), tetragonal/cubic-averaged constants.
"""
import json
import os
import sys

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import BFGS
from ase.units import GPa

from elastic import FMAX, STRAINS, make_calc, strained


def derived(M):
    S = (M + M.T) / 2
    d = np.diag(S)
    return dict(
        B_V=(d[:3].sum() + 2 * (S[0, 1] + S[0, 2] + S[1, 2])) / 9,
        G_V=(d[:3].sum() - (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * d[3:].sum()) / 15,
        min_eig=float(np.linalg.eigvalsh(S).min()),
        # Mandel/Kelvin (sqrt2 on shear) -> rotation-invariant eigenvalues; Voigt min_eig only has the right sign
        kelvin=np.linalg.eigvalsh(S * np.outer(m := np.r_[1, 1, 1, [2**0.5] * 3], m)).round(2).tolist(),
        C11=(d[0] + d[1]) / 2, C33=d[2], C12=S[0, 1], C13=(S[0, 2] + S[1, 2]) / 2,
        C44=(d[3] + d[4]) / 2, C66=d[5], asym=float(np.abs(M - M.T).max()), full=np.round(M, 2).tolist())


def tensor(a0, calc):
    a0.calc = calc
    BFGS(FrechetCellFilter(a0), logfile=None).run(fmax=FMAX, steps=1000)
    s0 = a0.get_stress()
    C = {"relaxed": np.zeros((6, 6)), "clamped": np.zeros((6, 6))}
    for j in range(6):
        ds = {"relaxed": [], "clamped": []}
        for e in STRAINS:
            a = strained(a0, j, e)
            a.calc = calc
            ds["clamped"].append(a.get_stress() - s0)
            BFGS(a, logfile=None).run(fmax=FMAX, steps=1000)
            ds["relaxed"].append(a.get_stress() - s0)
        for k in C:
            C[k][:, j] = np.polyfit(STRAINS, np.array(ds[k]), 1)[0] / GPa
    return a0, s0, {k: derived(M) for k, M in C.items()}


if __name__ == "__main__":
    kind, path, out = sys.argv[1:4]
    done = {json.loads(line)["name"] for line in open(out)} if os.path.exists(out) else set()
    calc = make_calc(kind, path)
    for spec in sys.argv[4:]:
        name, src = spec.split("=")
        if name in done:
            continue
        f, i = src.split("@")
        a0, s0, res = tensor(read(f, int(i)), calc)
        r = dict(name=name, model=os.path.basename(path.rstrip("/")), cellpar=a0.cell.cellpar().round(4).tolist(),
                 vol_per_atom=a0.get_volume() / len(a0), s0_GPa=float(np.abs(s0).max() / GPa), **res)
        print(name, r["cellpar"][:3], {k: round(v, 1) for k, v in res["relaxed"].items() if k in ("C11", "C33", "C12", "C13", "C44", "C66", "B_V")}, flush=True)
        open(out, "a").write(json.dumps(r) + "\n")
