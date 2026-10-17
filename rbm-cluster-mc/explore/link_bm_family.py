"""Wang 2017 link BM (Eqs. 3-5): rejection-free family in b. b -> -inf = Swendsen-Wang, finite b =
Niedermayer, b -> +inf = global flip. tau of symmetry-INVARIANT observables (|m|, E) per step.
Part 1: 2D Ising at T_c, L=8,16, 3 seeds. Part 2: 2D +-J glass L=8, T=0.7, 2 disorder x 2 chains."""
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice
from bm3 import link_bm_step, link_bm_weights
from run import tau_int

warnings.simplefilter("ignore")
BS = (-30, -3, -1, 0, 1, 3, 30)


def chain(lat, beta, W, b, n, rng):
    s = rng.integers(2, size=lat.N)
    for _ in range(500):
        s = link_bm_step(lat, beta, W, b, s, rng)[0]
    m, E, acc, ch, t0 = np.empty(n), np.empty(n), 0, 0, time.perf_counter()
    for t in range(n):
        s, a, c = link_bm_step(lat, beta, W, b, s, rng)
        m[t], E[t] = abs((2 * s - 1).mean()), lat.energy(s)
        acc += a; ch += c
    ms = 1e3 * (time.perf_counter() - t0) / n
    frozen = np.ptp(E) == 0                       # invariant observables never change
    return (np.inf if frozen else tau_int(m)), (np.inf if frozen else tau_int(E)), acc / n, ch / n / lat.N, ms


part = sys.argv[1] if len(sys.argv) > 1 else "ising"
if part == "ising":
    beta = 1 / 2.269185
    for L, n in ((8, 8000), (16, 10000)):
        lat = Lattice(L, 2)
        for b in BS:
            W = link_bm_weights(lat, beta, b)
            r = np.array([chain(lat, beta, W, b, n, np.random.default_rng(100 * L + k)) for k in range(3)])
            sem = lambda x: np.std(x) / np.sqrt(3) if np.all(np.isfinite(x)) else np.nan
            print(f"L={L:2d} b={b:4d} W={W[0]:.3f}  tau_|m|={r[:,0].mean():7.1f}+-{sem(r[:,0]):5.1f}  "
                  f"tau_E={r[:,1].mean():7.1f}+-{sem(r[:,1]):5.1f}  acc={r[:,2].mean():.3f}  "
                  f"changed/N={r[:,3].mean():.3f}  ms={r[:,4].mean():.2f}", flush=True)
else:
    beta, n = 1 / 0.7, 6000
    for b in BS:
        r = []
        for dis in range(2):
            lat = Lattice(8, 2, q=2, kind="pm", rng=100 + dis)   # same disorder as glass_gates.py
            W = link_bm_weights(lat, beta, b)
            r += [chain(lat, beta, W, b, n, np.random.default_rng(10 * dis + c)) for c in range(2)]
        r = np.array(r)
        print(f"glass b={b:4d}  tau_E={r[:,1].mean():7.1f} (per-run {np.round(r[:,1],1)})  acc={r[:,2].mean():.3f}  "
              f"changed/N={r[:,3].mean():.3f}  ms={r[:,4].mean():.2f}", flush=True)
