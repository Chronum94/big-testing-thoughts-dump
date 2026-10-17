"""Is KBD-BM at b=2 (T=0.5, L=8 FFIM) biased? Long chains, 4 seeds, vs transfer-matrix exact -0.998834.
Also reports acceptance and the fraction of steps that change any spin."""
import sys
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from bm3 import checkerboard, kbd_step, kbd_weights
from run import tau_int

warnings.simplefilter("ignore")
lat, beta, n = Lattice(8, 2, q=2, kind="ffim"), 2.0, 40000
cb = checkerboard(lat)
for name, b in (("KBD b=-30", -30.0), ("KBD b=2", 2.0), ("metropolis", None)):
    means, acc, moved, taus = [], 0, 0, []
    for seed in range(4):
        r = np.random.default_rng(seed)
        s = np.zeros(lat.N, int)                      # start near a ground state? no: random below
        s = r.integers(2, size=lat.N)
        E = np.empty(n)
        for t in range(n + 2000):
            if b is None:
                metropolis_sweep(lat, s, beta, r)
            else:
                s, a, c = kbd_step(lat, beta, kbd_weights(lat, beta, b), b, s, r, cb)
                if t >= 2000:
                    acc += a; moved += c > 0
            if t >= 2000:
                E[t - 2000] = lat.energy(s) / lat.N
        means.append(E.mean()); taus.append(tau_int(E))
    m, sem = np.mean(means), np.std(means, ddof=1) / 2
    print(f"{name:11s} <E>/N = {m:.5f} +- {sem:.5f}  (z vs exact = {(m + 0.998834) / sem:+.1f})  "
          f"tau_E={np.mean(taus):.1f}  acc={acc / (4 * n):.3f} moved={moved / (4 * n):.3f}  seeds={np.round(means, 5)}", flush=True)
