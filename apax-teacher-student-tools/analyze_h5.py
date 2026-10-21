"""Generic stability + transport analysis of apax MD runs (h5md).

usage: python analyze_h5.py --mobile K [--bonds Fe-C:2.3,C-N:1.4] RUN_DIR [RUN_DIR ...]
Per run: T mean/std, Epot drift, NVE drift (if the dir name has 'nve'), closest pair overall and per pair type,
for each --bonds X-Y:cut the fraction of X atoms whose coordination by Y within cut changed from the first frame,
and self-diffusion D (cm^2/s) of each species from the MSD (time origins after the first 20 %).
"""
import argparse
import glob

import h5py
import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.neighborlist import neighbor_list

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--bonds", default="")
a_ = ap.parse_args()
bonds = [(b.split(":")[0].split("-")[0], b.split(":")[0].split("-")[1], float(b.split(":")[1])) for b in a_.bonds.split(",") if b]
for d in sorted(r for pat in a_.runs for r in glob.glob(pat)):
    try:
        f = h5py.File(f"{d}/md.h5", "r"); cfg = yaml.safe_load(open(d + ".yaml"))
    except Exception as e:
        print(d, "unreadable:", e); continue
    p, o = f["particles/atoms"], f["observables/atoms"]; n = len(p["position/time"])
    if n < 5:
        print(d, f"only {n} frames"); continue
    ps = cfg["duration"] / 1000.0 * n / max(n, 1); t = np.linspace(0, cfg["duration"] / 1000.0, n)
    Z = p["species/value"][0].astype(int); sym = np.array([chemical_symbols[z] for z in Z]); N = len(Z)
    v = p["velocity/value"][:n]; ke = 0.5 * (atomic_masses[Z][None, :, None] * v**2).sum((1, 2)); T = 2 * ke / (3 * N * 8.617333e-5)
    ep = o["potential_energy/value"][:n].ravel(); s = n // 5
    line = f"{d}: {t[-1]:.1f} ps, T {T[s:].mean():.0f}+-{T[s:].std():.0f} K, Epot {1000 * (ep[-1] - ep[s]) / N:+.1f} meV/atom"
    if "nve" in d:
        line += f", NVE drift {1000 * np.polyfit(t, ep + ke, 1)[0] / N:+.4f} meV/atom/ps"
    boxes = p["box/edges/value"][:n]; R = p["position/value"][:n]
    mins, changed = {}, {b: 0.0 for b in bonds}
    ref_cn = {}
    for k in list(range(0, n, max(1, n // 30))):
        a = Atoms(numbers=Z, positions=R[k], cell=boxes[k], pbc=True); i, j, dd = neighbor_list("ijd", a, 3.0)
        for x, y, z in zip(sym[i], sym[j], dd):
            key = "-".join(sorted((x, y))); mins[key] = min(mins.get(key, 9), z)
        for b in bonds:
            m = (sym[i] == b[0]) & (sym[j] == b[1]) & (dd < b[2])
            cn = np.bincount(i[m], minlength=N)[sym == b[0]]
            if b not in ref_cn:
                ref_cn[b] = cn
            changed[b] = max(changed[b], float((cn != ref_cn[b]).mean()))
    line += " | closest " + " ".join(f"{k} {v_:.2f}" for k, v_ in sorted(mins.items()))
    if bonds:
        line += " | coordination changed (max over frames): " + " ".join(f"{b[0]}-{b[1]} {100 * changed[b]:.0f}%" for b in bonds)
    fr = np.einsum("tni,tij->tnj", R, np.linalg.inv(boxes)); dfr = np.diff(fr, axis=0); dfr -= np.round(dfr)
    un = np.einsum("tnj,jk->tnk", np.concatenate([fr[:1], fr[:1] + np.cumsum(dfr, axis=0)]), boxes.mean(0))
    lag = max(3, int(n * 0.25)); Ds = []
    for e in sorted(set(sym)):
        m = sym == e
        msd = np.array([((un[s + L:, m] - un[s:-L, m]) ** 2).sum(-1).mean() for L in range(1, lag)])
        tl = t[1:lag]; k0 = len(tl) // 3
        D = np.polyfit(tl[k0:], msd[k0:], 1)[0] / 6 * 1e-4 if len(tl) > k0 + 2 else np.nan
        Ds.append(f"{e} {D:.1e}")
    print(line + " | D (cm2/s) " + " ".join(Ds), flush=True)
