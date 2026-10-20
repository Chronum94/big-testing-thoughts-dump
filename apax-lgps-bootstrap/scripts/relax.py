"""Relax the lowest candidates (positions + cell) with a PET-MAD teacher; write sorted by relaxed energy.

usage: python relax.py CKPT cand.extxyz N out.extxyz
"""
import sys
import time

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from petjax import UPETCalculator

calc = UPETCalculator.from_checkpoint(sys.argv[1], stress=True)
cands = read(sys.argv[2], f":{int(sys.argv[3])}")
out = []
for a in cands:
    t0 = time.time(); a.calc = calc
    opt = FIRE(FrechetCellFilter(a), logfile=None)
    opt.run(fmax=0.01, steps=600)
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
    a.info["e_relax"] = e
    out.append(a)
    c = a.cell.cellpar()
    print(f"ge {a.info['ge']} li {a.info['li_id']}: E {e:.3f} eV ({opt.nsteps} steps, fmax {np.abs(f).max():.3f}) "
          f"a {c[0]:.3f} b {c[1]:.3f} c {c[2]:.3f} angles {c[3]:.1f} {c[4]:.1f} {c[5]:.1f} {time.time() - t0:.0f}s", flush=True)
out.sort(key=lambda x: x.info["e_relax"])
write(sys.argv[4], out)
