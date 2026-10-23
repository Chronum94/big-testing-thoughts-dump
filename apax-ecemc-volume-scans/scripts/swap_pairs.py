"""Carbonyl <-> ether rigid-swap pairs around Li+ (ordering test set; built from held-out s2 MD frames only).

usage: python swap_pairs.py OUT.extxyz MD1.extxyz [MD2 ...] [--max_pairs 150] [--seed 0] [--solvents EC,EMC,DMC] [--spins 12] [--min_contact 1.6]
For a free Li+ (single-atom molecule) and a solvent O bound to it (< 2.6 A), take an O of the other type in the same molecule
(carbonyl = O bonded to one C, ether = O bonded to two C), rigidly rotate the molecule about its COM so that O's direction
from the COM matches the bound O's, then translate it so the new O sits exactly at the bound O's position (same Li-O distance).
Then tries 12 spins of the swapped molecule about the Li-O axis and keeps the first without an intermolecular contact < 1.6 A. Writes both members of each pair (original first, unlabelled),
info: pair_id, member (0 = original, 1 = swapped), swap ('C->E' / 'E->C'), solvent, d_LiO.
"""
import argparse
import collections

import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("md", nargs="+")
ap.add_argument("--max_pairs", type=int, default=150); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--solvents", default="EC,EMC,DMC"); ap.add_argument("--spins", type=int, default=12); ap.add_argument("--min_contact", type=float, default=1.6)
a_ = ap.parse_args()
exec("import numpy as np\ndef whole" + open("/home/chronum/scratch/ecemc/volume_scan.py").read().split("def whole")[1].split("\n\n\n")[0])  # noqa: S102
NAMES = {"C3H4O3": "EC", "C4H8O3": "EMC", "C3H6O3": "DMC"}


def rot_onto(u, v):
    """Rotation matrix taking unit vector u onto unit vector v (Rodrigues)."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v); w = np.cross(u, v); s, c = np.linalg.norm(w), u @ v
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(np.cross(u, [1, 0, 0]), np.cross(u, [1, 0, 0])) / max(np.linalg.norm(np.cross(u, [1, 0, 0])) ** 2, 1e-12)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


rng = np.random.default_rng(a_.seed); frames = [(f, k, a) for f in a_.md for k, a in enumerate(read(f, ":", format="extxyz"))]
rng.shuffle(frames); out, pid, stats = [], 0, collections.Counter()
for f, k, a in frames:
    if pid >= a_.max_pairs:
        break
    mol, sym = a.arrays["mol_id"], np.array(a.get_chemical_symbols()); P = whole(a, mol); m = a.get_masses()
    d = a.get_all_distances(mic=True); C = a.cell.array; Ci = np.linalg.inv(C)
    lis = [i for i in range(len(a)) if sym[i] == "Li" and (mol == mol[i]).sum() == 1]
    for li in lis:
        for ob in np.where((sym == "O") & (d[li] < 2.6))[0]:
            g = mol[ob]; idx = np.where(mol == g)[0]; c = collections.Counter(sym[idx]); name = NAMES.get("".join(f"{e}{c[e]}" for e in sorted(c)))
            if name is None or name not in a_.solvents.split(","):
                continue
            nC = lambda o: sum(1 for j in idx if sym[j] == "C" and np.linalg.norm(P[j] - P[o]) < 1.6)  # noqa: E731
            tb = nC(ob); targets = [o for o in idx if sym[o] == "O" and o != ob and nC(o) != tb]
            if not targets:
                continue
            ot = targets[int(rng.integers(len(targets)))]
            com = (m[idx, None] * P[idx]).sum(0) / m[idx].sum(); R = rot_onto(P[ot] - com, P[ob] - com)
            Q0 = P.copy(); Q0[idx] = (P[idx] - com) @ R.T + com; Q0[idx] += P[ob] - Q0[ot]
            axis = P[ob] - P[li] - np.round((P[ob] - P[li]) @ Ci) @ C; axis /= np.linalg.norm(axis); ok = None
            for phi in np.arange(a_.spins) * 2 * np.pi / a_.spins:  # spin the swapped molecule about the Li-O axis (Li-O distance unchanged)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]); Rs = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * K @ K
                Q = Q0.copy(); Q[idx] = (Q0[idx] - P[ob]) @ Rs.T + P[ob]
                b = a.copy(); b.calc = None; b.positions = Q; b.wrap()
                db = b.get_all_distances(mic=True)
                if db[mol[:, None] != mol[None, :]].min() >= a_.min_contact:
                    ok = b; break
            if ok is None:
                stats["rejected (contact)"] += 1; continue
            b = ok
            kind = "C->E" if tb == 1 else "E->C"
            for mem, x in ((0, a.copy()), (1, b)):
                x.calc = None; x.info = {"pair_id": pid, "member": mem, "swap": kind, "solvent": name, "d_LiO": float(d[li, ob]), "src_frame": f"{f}:{k}", "src": "swap_pair"}
                x.arrays["mol_id"] = mol.copy(); out.append(x)
            pid += 1; stats[f"{name} {kind}"] += 1
            break  # one swap per Li per frame
        if pid >= a_.max_pairs:
            break
write(a_.out, out); print(f"{pid} pairs ({len(out)} frames) from {len(frames)} frames:", dict(stats))
