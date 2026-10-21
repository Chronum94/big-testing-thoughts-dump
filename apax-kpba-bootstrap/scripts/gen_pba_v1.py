"""PBA v1 blocks from the literature agent's findings (lit/PBA_PLAN.md):
  P  experimental P2_1/n K2Fe[Fe(CN)6] (32 atoms, K slid ~0.4 A along <110>, tilts): interpolation centred -> slid
     (0 to 1.5x), strains, small element-scaled rattles
  K  K potential-energy scans in cubic K8 / K4 cells: one K moved 0-1.4 A along <100>, <110>, <111>
  U  rigid-unit moves: Fe(CN)6 units translated (0.05-0.15 A); C-N pairs moved as a unit with a +-0.03 A stretch
     (the C-N bond only fluctuates ~0.03 A even at 900 K, so isotropic C/N rattles overshoot)
Labels: PET-MAD-m; max|F| <= 30 eV/A.

usage: python gen_pba_v1.py out.extxyz
"""
import itertools
import sys

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

rng = np.random.default_rng(31)
calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True, bucket_strategy="powers_of_2")
SCALE = {"K": 1.5, "Fe": 0.6, "C": 0.3, "N": 0.3, "Mn": 0.6}
frames = []


def add(a, cat, sigma=0.0, **info):
    b = a.copy(); b.calc = None
    if sigma:
        sc = np.array([SCALE.get(s, 1.0) for s in b.get_chemical_symbols()])
        b.positions += rng.normal(0, 1, b.positions.shape) * sigma * sc[:, None]
    b.info = {"category": cat, "n_K": int((b.symbols == "K").sum()), "rattle": sigma, **info}; frames.append(b)


# P: P2_1/n
lit = "/home/chronum/scratch/pba/lit"
cen, slid = read(f"{lit}/relaxed_m_p21n_K2Fe_K_centred.extxyz"), read(f"{lit}/relaxed_m_p21n_K2Fe_exp_slid.extxyz")
for f in np.linspace(0, 1.5, 7):
    b = cen.copy(); b.positions = cen.positions + f * (slid.positions - cen.positions)
    add(b, "P_p21n_slide", 0.0, slide=float(f)); add(b, "P_p21n_slide", 0.08, slide=float(f))
for e in (-0.03, -0.015, 0.015, 0.03):
    for ax in range(3):
        b = slid.copy(); m = np.eye(3); m[ax, ax] += e; b.set_cell(slid.cell.array @ m, scale_atoms=True)
        add(b, "P_p21n_strain", 0.05, strain=e, axis=ax)
for s in (0.05, 0.1, 0.15):
    for _ in range(4):
        add(slid, "P_p21n_rattle", s)

# K: scans
cubic = read("relaxed_pba.extxyz", ":")
dirs = {"100": np.array([1, 0, 0]), "110": np.array([1, 1, 0]) / 2**0.5, "111": np.array([1, 1, 1]) / 3**0.5}
for idx in (7, 3):  # K8, K4 alternating
    r = cubic[idx]; k = np.flatnonzero(r.symbols == "K")[0]
    for name, dvec in dirs.items():
        for dist in np.linspace(0.2, 1.4, 7):
            b = r.copy(); b.positions[k] += dist * dvec * rng.choice([-1, 1])
            add(b, "K_scan", 0.0, direction=name, dist=float(dist))

# U: rigid-unit moves
for r in cubic:
    s = np.array(r.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", r, 1.4)
    cn = [(a_, b_) for a_, b_ in zip(i, j) if s[a_] == "C" and s[b_] == "N"]
    for amp in (0.05, 0.10, 0.15):
        b = r.copy()  # translate every C-N pair rigidly, with a small stretch
        for c_, n_ in cn:
            t = rng.normal(0, amp, 3); b.positions[c_] += t; b.positions[n_] += t
            u = b.positions[n_] - b.positions[c_]; u /= np.linalg.norm(u); st = rng.normal(0, 0.015)
            b.positions[c_] -= st * u; b.positions[n_] += st * u
        b.positions[s == "K"] += rng.normal(0, 1.5 * amp, ((s == "K").sum(), 3))
        b.positions[s == "Fe"] += rng.normal(0, 0.5 * amp, ((s == "Fe").sum(), 3))
        add(b, "U_rigid_cn", 0.0, amp=amp)
print(f"generated {len(frames)}", flush=True)
out, dropped = [], 0
for a in frames:
    a.pbc = True; a.calc = calc
    e, f, st = a.get_potential_energy(), a.get_forces(), a.get_stress()
    if np.abs(f).max() > 30 or not np.isfinite(e):
        dropped += 1; continue
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=st); out.append(a)
write(sys.argv[1], out)
cats = [a.info["category"] for a in out]
print({c: cats.count(c) for c in dict.fromkeys(cats)}, f"dropped {dropped}, total {len(out)}")
