"""Phonon bands (ndsc nondiagonal supercells) for the cubic semiconductors, one model per call.

usage: phon.py teacher|apax PATH ELASTIC.jsonl OUT_DIR [names...]
Lattice constant = that model's own relaxed a from the elastic run (primitive 2-atom cell, forces zero by symmetry).
Writes OUT_DIR/NAME.npz: x, energies (THz), labels, Gamma/X/L frequencies; resumable.
ponytail: no LO-TO (no Born charges in teacher or students) -- polar compounds show the analytic, non-split TO=LO at Gamma.
"""
import json
import os
import sys

import numpy as np
from ase.build import bulk

sys.path.insert(0, "/home/chronum/scratch/elastic")
sys.path.insert(0, "/home/chronum/scratch/phon/ndsc")
from elastic import CUBIC, make_calc  # noqa: E402
from ndsc import phonons  # noqa: E402

MESH, PATH, NPTS = (6, 6, 6), "GXWKGL", 200
EV2THZ = 241.79893

kind, path, el, out = sys.argv[1:5]
A = {r["name"]: r["a"] for r in map(json.loads, open(el))}
names = sys.argv[5:] or list(CUBIC)
os.makedirs(out, exist_ok=True)
calc = make_calc(kind, path)
for n in names:
    f = f"{out}/{n}.npz"
    if os.path.exists(f):
        continue
    struct = CUBIC[n][0]
    prim = bulk(n, struct, a=A[n])  # primitive fcc-based cell
    prim.calc = calc
    fmax = np.abs(prim.get_forces()).max()
    # ponytail: PET is not exactly equivariant -> ~5 meV/A at symmetric sites; central differences cancel it
    assert fmax < 2e-2, (n, fmax)
    ph = phonons(prim, MESH, calc)
    ph.acoustic(ph.C_avNav)
    bp = prim.cell.bandpath(PATH, npoints=NPTS)
    bs = ph.get_band_structure(bp)
    E = bs.energies[0] * EV2THZ  # (nk, nbands)
    x, X, lab = bp.get_linear_kpoint_axis()
    sp = {lab_: E[np.argmin(np.abs(x - xx))] for xx, lab_ in zip(X, lab)}
    np.savez(f, x=x, X=X, labels=np.array(lab), E=E, G=sp["G"], Xp=sp["X"], L=sp["L"], a=A[n])
    print(n, f"a={A[n]:.4f}", "Gamma", np.round(sp["G"], 2), "X", np.round(sp["X"], 2), "min", round(E.min(), 3), flush=True)
