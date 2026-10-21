"""Superionic water bootstrap v0 (no MD), labelled with PET-MAD-m on the GPU.

Densities in g/cm3; the teacher's stress gives the pressure of every frame (stored as info P_GPa).
  V  ice VII (bcc O, 16 H2O, ice rules from random Eulerian orientations of the two diamond nets) and ice X (H at bond midpoints)
  S  superionic-like snapshots: bcc O (54 H2O) with H on random tetrahedral/octahedral interstices; fcc O (32 H2O) with
     H on all tetrahedral sites (antifluorite-like reference) or random tet/oct occupation; some relaxed 25 FIRE steps
  M  molecular-random: each O gets two H at 1.0 A in random directions (bcc 54)
  B  Bain path bcc -> fcc (c/a 1 -> sqrt 2 at fixed volume) with random interstitial H (bcc 54)
  L  liquid-like: random O (min O-O distance) with molecular-random H (54 H2O)
O and H are rattled separately on every frame; frames with an H-H or O-H distance < 0.7 A or max|F| > 40 eV/A are dropped.

usage: python gen_sice.py out.extxyz
"""
import itertools
import sys
import time

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import write  # noqa: E402
from ase.optimize import FIRE  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

rng = np.random.default_rng(11)
calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True, bucket_strategy="powers_of_2")
M_H2O = 18.015 / 0.60221  # g/mol -> g / (mol * 1e-24 cm3/A3): mass per molecule in g*A3/cm3 units


def box_len(n_mol, rho):
    return (n_mol * M_H2O / rho) ** (1 / 3)


def lattice(kind, n):
    """Fractional O positions of an n x n x n bcc (2/cell) or fcc (4/cell) supercell."""
    basis = {"bcc": [[0, 0, 0], [.5, .5, .5]], "fcc": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]}[kind]
    return np.array([(np.array(b) + c) / n for c in itertools.product(range(n), repeat=3) for b in basis])


def interstices(kind, n):
    """Fractional tetrahedral and octahedral interstitial sites of the same supercell."""
    if kind == "fcc":
        tet = [[x, y, z] for x in (.25, .75) for y in (.25, .75) for z in (.25, .75)]
        octs = [[.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]
    else:
        tet = [[.5, .25, 0], [.5, .75, 0], [0, .5, .25], [0, .5, .75], [.25, 0, .5], [.75, 0, .5]]
        tet += [list((np.array(t) + .5) % 1) for t in tet]
        octs = [[.5, .5, 0], [.5, 0, .5], [0, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]
    sh = list(itertools.product(range(n), repeat=3))
    return (np.array([(np.array(t) + c) / n for c in sh for t in tet]), np.array([(np.array(o) + c) / n for c in sh for o in octs]))


def pick_sites(cands, k, cell, dmin=1.0):
    """k random candidate sites (fractional), no two closer than dmin (minimum image)."""
    out = []
    for i in rng.permutation(len(cands)):
        p = cands[i]
        if all(np.linalg.norm((((p - q) + .5) % 1 - .5) @ cell) >= dmin for q in out):
            out.append(p)
        if len(out) == k:
            return np.array(out)
    return None


def assemble(o_frac, h_frac, L, c_over_a=1.0):
    cell = np.diag([L / c_over_a ** (1 / 3), L / c_over_a ** (1 / 3), L * c_over_a ** (2 / 3)])
    return Atoms(f"O{len(o_frac)}H{len(h_frac)}", scaled_positions=np.vstack([o_frac, h_frac]) % 1, cell=cell, pbc=True)


def molecular_h(o_cart, cell):
    h = []
    for o in o_cart:
        d = rng.normal(size=3); d /= np.linalg.norm(d); p = np.cross(d, rng.normal(size=3)); p /= np.linalg.norm(p)
        th = np.radians(rng.uniform(100, 110)) / 2
        h += [o + np.cos(th) * d + np.sin(th) * p, o + np.cos(th) * d - np.sin(th) * p]
    return np.array(h) @ np.linalg.inv(cell)


def ice_vii(L, symmetric):
    """16-molecule ice VII / X: two diamond nets on bcc O; bonds oriented by a random Eulerian circuit (ice rules)."""
    o = lattice("bcc", 2); cell = np.eye(3) * L
    sub = np.array([int((np.round(4 * p).astype(int).sum() // 2) % 2) for p in o])  # diamond sublattice label
    d = o[:, None] - o[None]; d -= np.round(d); dist = np.linalg.norm(d @ cell, axis=-1); np.fill_diagonal(dist, 9)
    adj = {i: [j for j in np.argsort(dist[i])[:8] if sub[j] == sub[i]] for i in range(len(o))}
    unused = {(min(i, j), max(i, j)) for i in adj for j in adj[i]}
    h = []
    # Hierholzer with random choices: each vertex has degree 4, so orienting along the circuit gives 2 in / 2 out
    while unused:
        start = next(iter(unused))[0]; stack, path = [start], []
        while stack:
            v = stack[-1]; nb = [w for w in adj[v] if (min(v, w), max(v, w)) in unused]
            if nb:
                w = nb[rng.integers(len(nb))]; unused.discard((min(v, w), max(v, w))); stack.append(w)
            else:
                path.append(stack.pop())
        for a_, b_ in zip(path[:-1], path[1:]):
            vec = (((o[b_] - o[a_]) + .5) % 1 - .5)
            frac = 0.5 if symmetric else 1.0 / np.linalg.norm(vec @ cell)  # O-H 1.0 A or midpoint
            h.append(o[a_] + frac * vec)
    return assemble(o, np.array(h), L)


frames = []


def add(a, cat, rho, sO, sH, **info):
    b = a.copy()
    o, hh = b.symbols == "O", b.symbols == "H"
    b.positions[o] += rng.normal(0, sO, (o.sum(), 3)); b.positions[hh] += rng.normal(0, sH, (hh.sum(), 3))
    d = b.get_all_distances(mic=True); np.fill_diagonal(d, 9)
    if d[hh].min() < 0.7:
        return False
    b.info = {"category": cat, "rho": rho, "rattle_O": sO, "rattle_H": sH, **info}
    frames.append(b)
    return True


def n_times(n, make, max_tries=40):
    """Call make() until it has returned True n times (or max_tries * n attempts)."""
    done = tries = 0
    while done < n and tries < max_tries * n:
        done += bool(make()); tries += 1
    return done


def relaxed(a, steps=25):
    b = a.copy(); b.calc = calc
    FIRE(b, logfile=None).run(fmax=0.1, steps=steps); b.calc = None
    return b


t0 = time.time()
RHO_BCC, RHO_FCC = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5), (3.0, 3.5, 4.0, 4.5, 5.0)
for rho in RHO_BCC:  # V: ice VII / X
    L = box_len(16, rho)
    for k in range(10):
        add(ice_vii(L, False), "V_iceVII", rho, 0.03 * (k % 3), 0.05 * (k % 3))
    for k in range(5):
        add(ice_vii(L, True), "V_iceX", rho, 0.02 * k, 0.04 * k)
for rho in RHO_BCC:  # S: bcc superionic-like
    L = box_len(54, rho); o = lattice("bcc", 3); tet, octs = interstices("bcc", 3); cell = np.eye(3) * L
    def bcc_random(relax_it):
        cands = np.vstack([tet, octs]) if rng.random() < 0.5 else tet
        h = pick_sites(cands + rng.normal(0, 0.1 / L, cands.shape), 108, cell, 0.8)
        if h is None:
            return False
        a = assemble(o, h, L)
        return add(relaxed(a) if relax_it else a, "S_bcc_relaxed" if relax_it else "S_bcc_random", rho, 0.02 if relax_it else 0.05, 0.05 if relax_it else 0.12)
    n_times(18, lambda: bcc_random(False)); n_times(6, lambda: bcc_random(True))
for rho in RHO_FCC:  # S: fcc
    L = box_len(32, rho); o = lattice("fcc", 2); tet, octs = interstices("fcc", 2); cell = np.eye(3) * L
    a_all = assemble(o, tet, L)
    for s in (0.0, 0.05, 0.1):
        add(a_all, "S_fcc_alltet", rho, s, 2 * s)
    add(relaxed(a_all), "S_fcc_alltet_relaxed", rho, 0.02, 0.05)
    def fcc_random(relax_it):
        cands = np.vstack([tet, octs])
        h = pick_sites(cands + rng.normal(0, 0.1 / L, cands.shape), 64, cell, 0.8)
        if h is None:
            return False
        a = assemble(o, h, L)
        return add(relaxed(a) if relax_it else a, "S_fcc_relaxed" if relax_it else "S_fcc_random", rho, 0.02 if relax_it else 0.05, 0.05 if relax_it else 0.12)
    n_times(18, lambda: fcc_random(False)); n_times(6, lambda: fcc_random(True))
for rho in (2.0, 2.5, 3.0, 3.5, 4.0):  # M: molecular-random on bcc O
    L = box_len(54, rho); o = lattice("bcc", 3); cell = np.eye(3) * L
    n_times(6, lambda: add(assemble(o, molecular_h(o @ cell, cell), L), "M_molecular", rho, 0.04, 0.04))
for rho in (2.5, 3.5, 4.5):  # B: Bain path
    L = box_len(54, rho); o = lattice("bcc", 3); tet, octs = interstices("bcc", 3)
    for ca in (1.0, 1.1, 1.2, 1.3, 2 ** 0.5):
        cell = assemble(o, tet[:1], L, ca).cell.array

        def bain():
            h = pick_sites(np.vstack([tet, octs]) + rng.normal(0, 0.1 / L, (len(tet) + len(octs), 3)), 108, cell, 0.8)
            return h is not None and add(assemble(o, h, L, ca), "B_bain", rho, 0.04, 0.1, c_over_a=ca)
        n_times(3, bain)
for rho in (1.5, 2.0, 2.5, 3.0, 3.5):  # L: liquid-like
    L = box_len(54, rho); cell = np.eye(3) * L; dmin = 0.8 * (M_H2O / rho) ** (1 / 3)
    def liquid(k):
        o = []
        while len(o) < 54:
            p = rng.random(3)
            if all(np.linalg.norm((((p - q) + .5) % 1 - .5) @ cell) >= dmin for q in o):
                o.append(p)
        o = np.array(o)
        a = assemble(o, molecular_h(o @ cell, cell), L)
        return add(a if k % 2 else relaxed(a), "L_liquid", rho, 0.03, 0.06)
    n_times(3, lambda: liquid(0), 15); n_times(3, lambda: liquid(1), 15)
# R: large rattles on a random subset of everything above (close O-H, H-H, O-O contacts)
for b in [frames[i] for i in rng.choice(len(frames), min(120, len(frames)), replace=False)]:
    n_times(1, lambda: add(b, "R_large_rattle", b.info["rho"], 0.15, 0.3), 10)
print(f"generated {len(frames)} frames {time.time() - t0:.0f}s", flush=True)

write(sys.argv[1] + ".raw.extxyz", frames)
out, dropped = [], 0
t1 = time.time()
for i, a in enumerate(frames):
    if i % 20 == 0:
        print(f"label {i}/{len(frames)} n_atoms {len(a)} {time.time() - t1:.0f}s", flush=True)
    a.calc = calc
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    if np.abs(f).max() > 40 or not np.isfinite(e):
        dropped += 1; continue
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
    a.info["P_GPa"] = float(-np.trace(s.reshape(3, 3)) / 3 * 160.2177 if s.size == 9 else -s[:3].mean() * 160.2177)
    out.append(a)
write(sys.argv[1], out)
cats = [a.info["category"] for a in out]
print({c: cats.count(c) for c in dict.fromkeys(cats)}, f"dropped {dropped}, total {len(out)}, {time.time() - t0:.0f}s")
for rho in sorted({a.info["rho"] for a in out}):
    P = [a.info["P_GPa"] for a in out if a.info["rho"] == rho]
    print(f"rho {rho}: P median {np.median(P):.0f} GPa (range {min(P):.0f}-{max(P):.0f}), n {len(P)}")
