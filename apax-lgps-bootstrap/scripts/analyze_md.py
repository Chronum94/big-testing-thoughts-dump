"""Stability + Li transport of LGPS student MD (apax h5md). usage: python analyze_md.py md/<run_dir> [...]"""
import sys

import h5py
import numpy as np
from ase import Atoms
from ase.data import atomic_masses
from ase.neighborlist import neighbor_list

for d in sys.argv[1:]:
    f = h5py.File(f"{d}/md.h5", "r"); p, o = f["particles/atoms"], f["observables/atoms"]
    n = len(p["position/time"]); t = p["position/time"][:] ; t = (t - t[0]) * (100.0 / t[-1] if "nvt" in d else 20.0 / t[-1])  # ps
    Z = p["species/value"][0].astype(int); N = len(Z); m = atomic_masses[Z]
    v = p["velocity/value"][:n]; T = (m[None, :, None] * v**2).sum((1, 2)) / (3 * N * 8.617333e-5)
    ep = o["potential_energy/value"][:n].ravel(); R = p["position/value"][:n]; box = p["box/edges/value"][0]
    sym = np.array(Atoms(numbers=Z).get_chemical_symbols()); li = sym == "Li"; cen = np.isin(sym, ["P", "Ge"])
    bad, mins, cn_bad = None, [], []
    for k in range(0, n, max(1, n // 40)):
        a = Atoms(numbers=Z, positions=R[k], cell=box, pbc=True)
        i, j, dd = neighbor_list("ijd", a, 3.0)
        lili = dd[li[i] & li[j]]
        cn = np.bincount(i[cen[i] & (sym[j] == "S") & (dd < 2.6)], minlength=N)[cen]
        mins.append((t[k], dd.min(), lili.min() if len(lili) else 9, (cn != 4).sum()))
        if bad is None and ((cn != 4).any() or dd.min() < 1.5):
            bad = t[k]
    mins = np.array(mins)
    # Li MSD from unwrapped positions (frames are wrapped: unwrap by minimum-image steps)
    fr = R[:, li] @ np.linalg.inv(box); dfr = np.diff(fr, axis=0); dfr -= np.round(dfr)
    un = np.concatenate([fr[:1], fr[:1] + np.cumsum(dfr, axis=0)]) @ box
    half = n // 2; msd = ((un[half:] - un[half]) ** 2).sum(-1).mean(-1)
    D = np.polyfit(t[half:] - t[half], msd, 1)[0] / 6 * 1e-4  # A^2/ps -> cm^2/s
    print(f"{d}: {t[-1]:.0f} ps, T {T[n // 10:].mean():.0f}+-{T[n // 10:].std():.0f} K, Epot drift {(ep[-1] - ep[n // 10]) / N * 1000:+.1f} meV/atom, "
          f"min pair {mins[:, 1].min():.2f} A, min Li-Li {mins[:, 2].min():.2f} A, broken tetrahedra frames {(mins[:, 3] > 0).sum()}/{len(mins)}"
          f"{f' (first at {bad:.1f} ps)' if bad is not None else ''}, D_Li(2nd half) {D:.2e} cm2/s, Li MSD end {msd[-1]:.1f} A2")
