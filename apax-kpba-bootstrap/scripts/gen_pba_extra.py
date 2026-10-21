"""PBA v0b: refill the rattle / volume blocks that v0 lost to the 30 eV/A force cut (the C-N bond is too stiff for
uniform 0.1-0.25 A rattles). Element-scaled rattles: K 1.5x, Fe 0.6x, C/N 0.4x of the base amplitude.

usage: python gen_pba_extra.py out.extxyz
"""
import sys

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

rng = np.random.default_rng(17)
calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True, bucket_strategy="powers_of_2")
SCALE = {"K": 1.5, "Fe": 0.6, "C": 0.4, "N": 0.4, "O": 0.6, "H": 1.0}
relaxed = read("relaxed_pba.extxyz", ":")
frames = []


def add(a, cat, sigma, **info):
    b = a.copy(); sc = np.array([SCALE[s] for s in b.get_chemical_symbols()])
    b.positions += rng.normal(0, 1, b.positions.shape) * sigma * sc[:, None]
    b.info = {"category": cat, "n_K": int((b.symbols == "K").sum()), "rattle": sigma, **info}; frames.append(b)


for r in relaxed:
    for s in (0.05, 0.10, 0.15, 0.20, 0.25):
        for _ in range(3):
            add(r, "D_rattle_scaled", s)
    for v in (0.85, 0.90, 0.95, 1.05, 1.10, 1.15):
        b = r.copy(); b.set_cell(r.cell.array * v ** (1 / 3), scale_atoms=True)
        for s in (0.05, 0.12):
            add(b, "D_volume_scaled", s, v_ratio=v)
out, dropped = [], 0
for a in frames:
    a.calc = calc
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    if np.abs(f).max() > 30 or not np.isfinite(e):
        dropped += 1; continue
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s); out.append(a)
write(sys.argv[1], out)
cats = [a.info["category"] for a in out]
print({c: cats.count(c) for c in dict.fromkeys(cats)}, f"dropped {dropped}, total {len(out)}")
