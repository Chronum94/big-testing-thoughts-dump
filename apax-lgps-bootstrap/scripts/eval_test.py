"""Score LGPS students on val + teacher-MD test frames (CPU), per category. usage: python eval_test.py run [run ...]"""
import sys

sys.path.insert(0, "/home/chronum/scratch/lgps")
import cpuload  # noqa: E402,F401  (GPU checkpoints -> CPU)
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from apax.md import ASECalculator  # noqa: E402

sets = {"val": read("data/val.extxyz", ":"), "test": read("data/test.extxyz", ":")}
for r in sys.argv[1:]:
    calc = ASECalculator(f"models/{r}")
    for name, frames in sets.items():
        rows = {}
        for a in frames:
            E, F = a.get_potential_energy(), a.get_forces()
            b = a.copy(); b.calc = calc
            rows.setdefault(a.info["category"], []).append(
                (abs(b.get_potential_energy() - E) / len(a), np.abs(b.get_forces() - F).mean()))
        for c, v in sorted(rows.items()):
            v = np.array(v)
            print(f"{r:14s} {name:4s} {c:16s} n {len(v):3d}  E {1000 * v[:, 0].mean():6.2f} meV/atom  F {1000 * v[:, 1].mean():6.1f} meV/A", flush=True)
