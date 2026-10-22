"""Rigid-molecule volume scans (Magdau et al. 2023 style): rescale the box and molecular centres of mass by s while every
molecule keeps its internal geometry, so only intermolecular distances change.

usage: python volume_scan.py OUT.extxyz MD1.extxyz [MD2 ...] [--every K] [--scales 0.95,0.98,1.02,1.05,1.10] [--affine]
--affine: plain ASE-style affine scaling (set_cell(scale_atoms=True): every bond stretches too), the naive control.
--aniso NORM: rigid molecules, but centres of mass and cell move by U = expm(ln(s) I + D), D random traceless with Frobenius
  norm exactly NORM, so det U = s^3 (same volume change as the isotropic scan). --aniso_mode ortho: D diagonal (stretch only,
  cell stays orthorhombic); general: D symmetric with off-diagonals (stretch + shear, triclinic cell).
Takes every K-th frame of each teacher MD file (needs arrays['mol_id']); writes unlabelled frames with info scan_s and
src_frame for the teacher to label. Skips scaled frames whose closest intermolecular contact drops below 1.5 A.
"""
import argparse

import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("md", nargs="+")
ap.add_argument("--every", type=int, default=8); ap.add_argument("--scales", default="0.95,0.98,1.02,1.05,1.10"); ap.add_argument("--affine", action="store_true"); ap.add_argument("--aniso", type=float, default=0.0); ap.add_argument("--aniso_mode", choices=["ortho", "general"], default="general"); ap.add_argument("--seed", type=int, default=0)
a_ = ap.parse_args(); scales = [float(x) for x in a_.scales.split(",")]


def whole(a, mol, rbond=1.9):
    """Unwrap each molecule along its bond graph (BFS, minimum image per bond < rbond A). Unwrapping around one atom fails
    when a molecule is longer than half the box (EMC in an 11 A box)."""
    P = a.positions.copy(); C = a.cell.array; Ci = np.linalg.inv(C)
    mic = lambda d: d - np.round(d @ Ci) @ C  # noqa: E731
    for m in np.unique(mol):
        idx = np.where(mol == m)[0]; placed = {idx[0]}; todo = [idx[0]]
        while todo:
            i = todo.pop()
            for j in idx:
                if j not in placed:
                    d = mic(P[j] - P[i])
                    if np.linalg.norm(d) < rbond:
                        P[j] = P[i] + d; placed.add(j); todo.append(j)
        for j in idx:  # dissociated pieces (LiF / HF can split in MD): place by minimum image to the first atom
            if j not in placed:
                P[j] = P[idx[0]] + mic(P[j] - P[idx[0]])
    return P


from scipy.linalg import expm  # noqa: E402
rng = np.random.default_rng(a_.seed)
out, skipped = [], 0
for f in a_.md:
    for k, a in enumerate(read(f, ":", format="extxyz")):
        if k % a_.every:
            continue
        mol = a.arrays["mol_id"]; P = whole(a, mol); m = a.get_masses()
        com = {g: (m[mol == g, None] * P[mol == g]).sum(0) / m[mol == g].sum() for g in np.unique(mol)}
        for s in scales:
            b = a.copy(); b.calc = None
            Q = P.copy()
            for g, c in com.items():
                Q[mol == g] += (s - 1) * c
            if a_.affine:
                b.set_cell(a.cell * s, scale_atoms=True)
            elif a_.aniso > 0:
                G = rng.normal(size=(3, 3)); D = np.diag(np.diag(G)) if a_.aniso_mode == "ortho" else 0.5 * (G + G.T)
                D -= np.trace(D) / 3 * np.eye(3); D *= a_.aniso / np.linalg.norm(D)
                U = expm(np.log(s) * np.eye(3) + D)  # det U = s^3
                Q = P.copy()
                for g, c in com.items():
                    Q[mol == g] += c @ U.T - c
                b.set_cell(a.cell.array @ U.T, scale_atoms=False); b.positions = Q
            else:
                b.set_cell(a.cell * s, scale_atoms=False); b.positions = Q
            b.wrap()
            d = b.get_all_distances(mic=True); inter = d[mol[:, None] != mol[None, :]].min()
            if inter < 1.5 and not a_.affine:
                skipped += 1; continue
            b.info = {k2: v for k2, v in a.info.items() if k2 in ("comp", "rho", "seed", "T")}
            b.info.update(scan_s=s, src_frame=f"{f}:{k}", src="affine_scan" if a_.affine else (f"aniso_{a_.aniso_mode}_scan" if a_.aniso > 0 else "volume_scan")); out.append(b)
write(a_.out, out)
print(f"{len(out)} scan frames from {len(a_.md)} files ({skipped} skipped: intermolecular contact < 1.5 A)")
