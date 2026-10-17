"""Wang App. B (no numerics in the paper): KBD-family BM on the fully frustrated Ising model (Villain).
Arms: Metropolis sweep, Swendsen-Wang (link BM b=-30), KBD-BM for several b (b=-30 is KBD).
Z2-invariant observables only: tau_E, and mean tau of bond variables J s_i s_j. 2 seeds; tau x ms/step."""
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from bm3 import bond_products, checkerboard, kbd_step, kbd_weights, link_bm_step, link_bm_weights
from run import tau_int

warnings.simplefilter("ignore")
kind = sys.argv[1] if len(sys.argv) > 1 else "ffim"
n = 8000
for T in (1.0, 0.5):
    beta = 1 / T
    for L in (8, 16):
        lat = Lattice(L, 2, q=2, kind=kind, rng=200)
        cb = checkerboard(lat)
        arms = {"metropolis": lambda s, r: (metropolis_sweep(lat, s, beta, r), s)[1],
                "SW": lambda s, r, W=link_bm_weights(lat, beta, -30.0): link_bm_step(lat, beta, W, -30.0, s, r)[0]}
        for b in (-30.0, -2.0, 0.0, 2.0):
            arms[f"KBD-BM b={b:g}"] = lambda s, r, b=b, W=kbd_weights(lat, beta, b): kbd_step(lat, beta, W, b, s, r, cb)[0]
        for name, step in arms.items():
            res = []
            for seed in range(2):
                r = np.random.default_rng(seed)
                s = r.integers(2, size=lat.N)
                for _ in range(1000):
                    s = step(s, r)
                E, B, t0 = np.empty(n), np.empty((n, len(lat.J)), np.int8), time.perf_counter()
                for t in range(n):
                    s = step(s, r)
                    B[t] = np.sign(lat.J) * bond_products(lat, s)
                    E[t] = -(lat.J * bond_products(lat, s)).sum()
                ms = 1e3 * (time.perf_counter() - t0) / n
                tb = np.nanmean([tau_int(B[:, k]) for k in range(B.shape[1])])
                res.append((tau_int(E), tb, ms, E.mean() / lat.N))
            r_ = np.array(res).mean(0)
            print(f"{kind} T={T} L={L:2d} {name:16s} tau_E={r_[0]:7.1f} tau_bond={r_[1]:7.1f} ms={r_[2]:5.2f} "
                  f"tauB*ms={r_[1]*r_[2]:8.1f}  <E>/N={r_[3]:.4f} (seeds {res[0][3]:.4f},{res[1][3]:.4f})", flush=True)
