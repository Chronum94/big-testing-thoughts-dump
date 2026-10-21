"""K_x Fe[Fe(CN)6] (K-PBA) bootstrap v0, labelled with PET-MAD-m on the GPU. No MD frames.

Cubic Fm-3m conventional cell (Fe4[Fe(CN)6]4, 56 framework atoms) + 0-8 K on the 8c cage centres.
  R  relaxed cells (positions + cell): 2 K arrangements per K count 0/2/4/6/8
  A  elastic: relaxed x 11 strains x rattle 0.03 / 0.07 A
  B  K configurations: random K arrangements per count (20 FIRE steps); K off-centre (0.3-1.2 A); K hop paths
     through the cage window (8c -> face centre -> 8c), unrelaxed and with the framework relaxed around a fixed K
  C  defects: one [Fe(CN)6] vacancy (K 0/2/4), CN flips (C<->N swapped on one linker)
  D  octahedral tilts (rigid rotation of each Fe(CN)6 unit, 2-8 deg), large rattles 0.10-0.25 A, volume scan 0.85-1.15
  E  water: zeolitic H2O in empty cages, coordinated H2O at a vacancy (few frames: the model has seen water)
Frames with max|F| > 30 eV/A are dropped.

usage: python gen_pba.py out.extxyz
"""
import itertools
import sys
import time

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.io import write  # noqa: E402
from ase.optimize import FIRE  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

rng = np.random.default_rng(7)
calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True)
A0, XC, XN = 10.2, 0.19, 0.19 + 1.15 / 10.2
FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
AX = np.vstack([np.eye(3), -np.eye(3)])
K8C = np.array(list(itertools.product([.25, .75], repeat=3)))


def framework():
    sym, pos = [], []
    for f in FCC:
        sym.append("Fe"); pos.append(f)               # Fe(C), low-spin side, 4a
        sym.append("Fe"); pos.append(f + [.5, 0, 0])  # Fe(N), 4b
        for d in AX:
            sym.append("C"); pos.append(f + XC * d)
            sym.append("N"); pos.append(f + XN * d)
    a = Atoms(sym, scaled_positions=np.mod(pos, 1), cell=np.eye(3) * A0, pbc=True)
    a.arrays["unit"] = np.repeat(np.arange(4), 14)  # Fe(C) unit id for tilts: Fe(C), Fe(N), 6 C, 6 N per block
    return a


def with_k(base, sites):
    a = base.copy()
    for s in sites:
        a += Atoms("K", scaled_positions=[K8C[s]], cell=a.cell)
    a.arrays["unit"] = np.concatenate([base.arrays["unit"], -np.ones(len(sites), int)])
    return a


def k_sites(r):
    """Indices into K8C of the sites holding K (nearest site, minimum image)."""
    out = []
    for p in r.get_scaled_positions()[r.symbols == "K"]:
        d = ((K8C - p) + .5) % 1 - .5
        out.append(int(np.linalg.norm(d @ r.cell.array, axis=1).argmin()))
    return out


def mic(r, vecs):
    """Minimum-image Cartesian vectors."""
    f = vecs @ np.linalg.inv(r.cell.array); f -= np.round(f)
    return f @ r.cell.array


def relax(a, cell=True, steps=600, fmax=0.01, fixed=None):
    a = a.copy(); a.calc = calc
    if fixed is not None:
        a.set_constraint(FixAtoms(indices=fixed))
    FIRE(FrechetCellFilter(a) if cell else a, logfile=None).run(fmax=fmax, steps=steps)
    a.set_constraint(); a.calc = None
    return a


frames = []


def add(a, cat, sigma=0.0, **info):
    b = a.copy()
    if sigma:
        b.positions += rng.normal(0, sigma, b.positions.shape)
    b.info = {"category": cat, "n_K": int((b.symbols == "K").sum()), "rattle": sigma, **info}
    frames.append(b)


t0 = time.time()
base = framework()
# R: relaxed cells
relaxed = []
for nk in (0, 2, 4, 6, 8):
    choices = {0: [()], 2: [(0, 7)], 4: [(0, 3, 5, 6), (0, 1, 2, 3)], 6: [(0, 1, 2, 3, 4, 5)], 8: [tuple(range(8))]}[nk]
    if nk in (2, 6):
        choices.append(tuple(int(x) for x in sorted(rng.choice(8, nk, replace=False))))
    for c in choices:
        r = relax(with_k(base, c)); r.info["k_sites"] = str(c); r.info["n_K"] = nk; relaxed.append(r)
        print(f"relaxed K{nk} {c}: a/b/c {r.cell.cellpar()[:3].round(3)}", flush=True)
write("relaxed_pba.extxyz", relaxed)
for r in relaxed:
    add(r, "R_relaxed")
print(f"R done {time.time() - t0:.0f}s", flush=True)

# A: elastic
strains = [np.eye(3)] + [np.eye(3) * (1 + s) for s in (-0.04, -0.02, 0.02, 0.04)]
for ax_ in (0, 2):
    for s in (-0.03, 0.03):
        e = np.eye(3); e[ax_, ax_] += s; strains.append(e)
for (i, j) in ((0, 1), (0, 2)):
    e = np.eye(3); e[i, j] = e[j, i] = 0.02; strains.append(e)
for r in relaxed:
    for e in strains:
        b = r.copy(); b.set_cell(r.cell.array @ e, scale_atoms=True)
        for s in (0.03, 0.07):
            add(b, "A_elastic", s)

# B: K configurations, off-centre K, hop paths
cell_by_nk = {}
for r in relaxed:
    cell_by_nk.setdefault(r.info["n_K"] if "n_K" in r.info else int((r.symbols == "K").sum()), r.cell.array)
fw = relax(base)  # relaxed empty framework as the template for K placement
for nk in (1, 2, 3, 4, 5, 6, 7):
    combos = list(itertools.combinations(range(8), nk)); rng.shuffle(combos)
    for c in combos[:12]:
        a = with_k(fw, c); a.set_cell(cell_by_nk.get(nk, cell_by_nk[min(cell_by_nk, key=lambda k: abs(k - nk))]), scale_atoms=True)
        add(relax(a, cell=False, steps=20, fmax=0.05), "B_K_config", k_sites=str(c))
for r in relaxed:
    k = np.flatnonzero(r.symbols == "K")
    for _ in range(5 if len(k) else 0):
        b = r.copy(); m = rng.choice(k, max(1, len(k) // 2), replace=False)
        d = rng.normal(size=(len(m), 3)); d /= np.linalg.norm(d, axis=1)[:, None]
        b.positions[m] += d * rng.uniform(0.3, 1.2, (len(m), 1))
        add(b, "B_K_offcentre", 0.02)
for r in relaxed:
    k = np.flatnonzero(r.symbols == "K")
    if not len(k) or len(k) == 8:
        continue
    empty = [K8C[s] for s in range(8) if s not in k_sites(r)]
    i = k[0]; start = r.get_scaled_positions()[i]
    target = min(empty, key=lambda s: np.linalg.norm(((s - start) + .5) % 1 - .5))
    dv = ((target - start) + .5) % 1 - .5
    for f in np.linspace(0, 1, 9):
        b = r.copy(); sp = b.get_scaled_positions(); sp[i] = start + f * dv; b.set_scaled_positions(sp)
        add(b, "B_K_hop", 0.0, hop_frac=float(f))
        if f in (0.25, 0.5, 0.75):
            add(relax(b, cell=False, steps=15, fmax=0.05, fixed=[i]), "B_K_hop_relaxed", 0.0, hop_frac=float(f))
print(f"B done {time.time() - t0:.0f}s", flush=True)

# C: [Fe(CN)6] vacancy, CN flips
vac_fw = base[[j for j in range(len(base)) if not (base.arrays["unit"][j] == 0 and j % 14 != 1)]]
for nk in (0, 2, 4):
    for _ in range(3):
        c = tuple(sorted(rng.choice(8, nk, replace=False)))
        a = vac_fw.copy()
        for s in c:
            a += Atoms("K", scaled_positions=[K8C[s]], cell=a.cell)
        a.set_cell(cell_by_nk[nk], scale_atoms=True)
        a = relax(a, cell=False, steps=40, fmax=0.05)
        for s in (0.0, 0.05, 0.1):
            add(a, "C_vacancy", s)
for r in relaxed:
    for _ in range(2):
        b = r.copy(); u = rng.integers(4); d = rng.integers(6); j0 = 14 * u + 2 + 2 * d  # C index; N is j0 + 1
        sym = b.get_chemical_symbols(); sym[j0], sym[j0 + 1] = sym[j0 + 1], sym[j0]; b.set_chemical_symbols(sym)
        add(b, "C_cn_flip", 0.02)

# D: tilts, large rattles, volume
for r in relaxed:
    for ang in (2, 4, 6, 8):
        b = r.copy()
        for u in range(4):
            m = np.flatnonzero(r.arrays["unit"] == u); m = m[(m % 14) != 1]  # Fe(C) + its 6 C + 6 N
            ctr = r.positions[14 * u]; rel = mic(r, r.positions[m] - ctr); ax_ = rng.normal(size=3); ax_ /= np.linalg.norm(ax_)
            th = np.radians(ang) * rng.choice([-1, 1]); K_ = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
            Rm = np.eye(3) + np.sin(th) * K_ + (1 - np.cos(th)) * K_ @ K_
            b.positions[m] = rel @ Rm.T + ctr
        add(b, "D_tilt", 0.02, tilt_deg=ang)
    for s in (0.10, 0.15, 0.20, 0.25):
        for _ in range(2):
            add(r, "D_rattle", s)
    for v in (0.85, 0.90, 0.95, 1.05, 1.10, 1.15):
        b = r.copy(); b.set_cell(r.cell.array * v ** (1 / 3), scale_atoms=True)
        add(b, "D_volume", 0.05, v_ratio=v); add(b, "D_volume", 0.15, v_ratio=v)

# E: water
def h2o(center):
    d = rng.normal(size=3); d /= np.linalg.norm(d); p = np.cross(d, rng.normal(size=3)); p /= np.linalg.norm(p)
    th = np.radians(104.5) / 2
    return Atoms("OH2", positions=[center, center + 0.97 * (np.cos(th) * d + np.sin(th) * p), center + 0.97 * (np.cos(th) * d - np.sin(th) * p)])
for nk in (0, 2, 4):
    r = next(x for x in relaxed if (x.symbols == "K").sum() == nk)
    empty = [K8C[s] for s in range(8) if s not in k_sites(r)]
    for nw in (1, 2, 4):
        if nw > len(empty):
            continue
        for _ in range(3):
            b = r.copy()
            for s in rng.permutation(len(empty))[:nw]:
                b += h2o(empty[s] @ r.cell.array)
            add(relax(b, cell=False, steps=30, fmax=0.05), "E_water_zeolitic", 0.03, n_H2O=nw)
for _ in range(4):  # coordinated water at a vacancy: O where the vacancy's N were (Fe(N)-O ~2.1 A)
    a = vac_fw.copy(); a.set_cell(cell_by_nk[0], scale_atoms=True)
    for d in AX:
        a += h2o((XN * d) @ a.cell.array)  # O on the former N sites around the removed Fe(C) at the origin
    add(relax(a, cell=False, steps=40, fmax=0.05), "E_water_coordinated", 0.03, n_H2O=6)
print(f"generated {len(frames)} frames {time.time() - t0:.0f}s", flush=True)

out, dropped = [], 0
for a in frames:
    a.pbc = True; a.calc = calc
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    if np.abs(f).max() > 30 or not np.isfinite(e):
        dropped += 1; continue
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
    a.arrays.pop("unit", None); out.append(a)
write(sys.argv[1], out)
cats = [a.info["category"] for a in out]
print({c: cats.count(c) for c in dict.fromkeys(cats)}, f"dropped {dropped}, total {len(out)}, {time.time() - t0:.0f}s")
