"""NPT analysis: volume per 50-atom cell vs COD, Li Wyckoff-site occupancies (Kuhn 2013 labels), D_c / D_ab.

usage: python analyze_npt.py md/<run_dir> PS [...]  (PS = simulated time of that run in ps)
"""
import sys

import h5py
import numpy as np

sys.path.insert(0, "/home/chronum/scratch/lgps")
from sites import expand  # noqa: E402

full, lab, _ = expand(); lab = lab.astype(str)
li_sites = np.char.startswith(lab, "Li")
site_frac = full.get_scaled_positions()[li_sites]; site_lab = lab[li_sites]  # 32 sites in the 1x1x1 cell
REP = 2  # the MD cell is 2x2x2
shifts = np.array([[i, j, k] for i in range(REP) for j in range(REP) for k in range(REP)])
sc_frac = ((site_frac[None] + shifts[:, None]) / REP).reshape(-1, 3); sc_lab = np.tile(site_lab, len(shifts))
KUHN = {"Li1": 0.466, "Li2": 0.86, "Li3": 0.74, "Li4": 0.81}
COD = {100: 8.6521**2 * 12.5816, 298: 8.7187**2 * 12.6385}  # A^3 per 50-atom cell
args = sys.argv[1:]
for d, ps in zip(args[::2], map(float, args[1::2])):
    f = h5py.File(f"{d}/md.h5", "r"); p = f["particles/atoms"]; n = len(p["position/time"]); s = n // 5  # drop first 20 %
    Z = p["species/value"][0].astype(int); li = Z == 3
    boxes = p["box/edges/value"][:n]; V = np.abs(np.linalg.det(boxes)) / REP**3
    R = p["position/value"][:n]
    # site occupancies: each Li to its nearest site (minimum image, Cartesian), counted if within 1.0 A
    occ = {k: 0.0 for k in KUHN}; unassigned = 0.0; frames = range(s, n, max(1, (n - s) // 400))
    for k in frames:
        fr = R[k, li] @ np.linalg.inv(boxes[k])
        dfr = fr[:, None] - sc_frac[None]; dfr -= np.round(dfr)
        dist = np.linalg.norm(dfr @ boxes[k], axis=-1); j = dist.argmin(1); ok = dist[np.arange(len(j)), j] < 1.0
        for name in KUHN:
            occ[name] += np.sum(ok & (sc_lab[j] == name))
        unassigned += np.sum(~ok)
    nf = len(frames)
    occ = {k: v / nf / np.sum(sc_lab == k) for k, v in occ.items()}
    # D_c, D_ab from unwrapped fractional positions (box varies: unwrap in fractional space)
    fr = np.einsum("tni,tij->tnj", R[:, li], np.linalg.inv(boxes)); dfr = np.diff(fr, axis=0); dfr -= np.round(dfr)
    un = np.einsum("tnj,jk->tnk", np.concatenate([fr[:1], fr[:1] + np.cumsum(dfr, axis=0)]), boxes[s:].mean(0))
    t = np.linspace(0, ps, n); lag = int(n * 0.2)
    msd = np.array([((un[s + L:] - un[s:-L]) ** 2).mean((0, 1)) for L in range(1, lag)]); tl = t[1:lag]; k0 = len(tl) // 4
    sl = [np.polyfit(tl[k0:], msd[k0:, a], 1)[0] for a in range(3)]
    Dab, Dc = (sl[0] + sl[1]) / 4 * 1e-4, sl[2] / 2 * 1e-4
    print(f"{d}: V {V[s:].mean():.1f} +- {V[s:].std():.1f} A3/cell (COD 100 K {COD[100]:.1f}, 298 K {COD[298]:.1f}) | "
          + " ".join(f"{k} {v:.3f} ({KUHN[k]})" for k, v in occ.items())
          + f" | off-site {unassigned / nf / li.sum():.3f} | D_c {Dc:.2e} D_ab {Dab:.2e} cm2/s, c/ab {Dc / Dab:.1f} | Li MSD/lag-max {msd[-1].sum():.1f} A2")
