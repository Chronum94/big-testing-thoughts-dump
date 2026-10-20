"""LGPS bootstrap set, 1x1x1 cells (50 atoms), labelled with PET-MAD-m.

A  equilibrium + elastic: every relaxed ordering x 11 strains x rattle 0.03 / 0.07 A
B  Li disorder: 50 Li orderings per Ge arrangement (all 6), 20 fixed-cell FIRE steps; plus a third of the starts
D  large rattles 0.10-0.25 A on relaxed + B structures; volume scan V/V0 0.80-1.15 x rattle 0.05 / 0.15 A
Frames with max|F| > 30 eV/A are dropped. Every frame carries category, ge, li_id.

usage: python gen_boot.py out.extxyz
"""
import sys
import time

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.optimize import FIRE  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

rng = np.random.default_rng(1)
calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True)
relaxed = read("relaxed_m.extxyz", ":")
cands = read("cand_s.extxyz", ":")
t0 = time.time()

# the Ge arrangement (2,3) has no relaxed cell yet: relax its lowest candidate
have = {a.info["ge"] for a in relaxed}
for ge in sorted({a.info["ge"] for a in cands} - have):
    a = min((c for c in cands if c.info["ge"] == ge), key=lambda c: c.info["e_s"]).copy()
    a.calc = calc
    FIRE(FrechetCellFilter(a), logfile=None).run(fmax=0.01, steps=600)
    a.info["e_relax"] = a.get_potential_energy(); a.calc = None
    relaxed.append(a)
    print(f"relaxed ge {ge}: E {a.info['e_relax']:.3f}", flush=True)
write("relaxed_all.extxyz", relaxed)
cell0 = np.mean([a.cell.array for a in relaxed], axis=0)

frames = []


def add(a, cat, sigma=0.0):
    b = a.copy()
    if sigma:
        b.positions += rng.normal(0, sigma, b.positions.shape)
    b.info = {k: a.info[k] for k in ("ge", "li_id") if k in a.info}
    b.info["category"], b.info["rattle"] = cat, sigma
    frames.append(b)


# A
strains = [np.eye(3)] + [np.eye(3) * (1 + s) for s in (-0.04, -0.02, 0.02, 0.04)]
for ax in (0, 2):
    for s in (-0.03, 0.03):
        e = np.eye(3); e[ax, ax] += s; strains.append(e)
for (i, j) in ((0, 1), (0, 2)):
    e = np.eye(3); e[i, j] = e[j, i] = 0.02; strains.append(e)
for a in relaxed:
    for e in strains:
        b = a.copy(); b.set_cell(a.cell.array @ e, scale_atoms=True)
        for s in (0.03, 0.07):
            add(b, "A_elastic", s)

# B
used = {(a.info["ge"], a.info["li_id"]) for a in relaxed}
b_structs = []
for ge in sorted({a.info["ge"] for a in cands}):
    pool = [c for c in cands if c.info["ge"] == ge and (ge, c.info["li_id"]) not in used]
    for k in rng.choice(len(pool), 50, replace=False):
        a = pool[k].copy(); a.set_cell(cell0, scale_atoms=True)
        if rng.random() < 1 / 3:
            add(a, "B_li_start")
        a.calc = calc
        FIRE(a, logfile=None).run(fmax=0.05, steps=20)
        a.calc = None
        add(a, "B_li_disorder"); b_structs.append(a)
print(f"B done {time.time() - t0:.0f}s", flush=True)

# D
for a in relaxed:
    for s in (0.10, 0.15, 0.20, 0.25):
        for _ in range(3):
            add(a, "D_rattle", s)
for k in rng.choice(len(b_structs), 60, replace=False):
    add(b_structs[k], "D_rattle", float(rng.choice([0.10, 0.15, 0.20, 0.25])))
for a in relaxed:
    for v in (0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15):
        b = a.copy(); b.set_cell(a.cell.array * v ** (1 / 3), scale_atoms=True)
        for s in (0.05, 0.15):
            add(b, "D_volume", s)
        frames[-1].info["v_ratio"] = frames[-2].info["v_ratio"] = v

# label
out, dropped = [], 0
for a in frames:
    a.pbc = True; a.calc = calc
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    if np.abs(f).max() > 30:
        dropped += 1; continue
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
    out.append(a)
write(sys.argv[1], out)
cats = [a.info["category"] for a in out]
print({c: cats.count(c) for c in dict.fromkeys(cats)}, f"dropped {dropped}, total {len(out)}, {time.time() - t0:.0f}s")
