"""FFIM L=16: tau of the structure-factor peaks S(0,0), S(0,pi) (Z2-invariant slow modes) and E,
for Metropolis sweeps, KBD (b=-30), and a mixture (1 sweep + 1 KBD step). 2 seeds; tau x ms/step."""
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from bm3 import checkerboard, kbd_step, kbd_weights
from run import tau_int

warnings.simplefilter("ignore")
L, n = 16, 20000
lat = Lattice(L, 2, q=2, kind="ffim")
cb = checkerboard(lat)
for T in (0.5, 0.3):
    beta = 1 / T
    W = kbd_weights(lat, beta, -30.0)
    kbd = lambda s, r: kbd_step(lat, beta, W, -30.0, s, r, cb)[0]
    sweep = lambda s, r: (metropolis_sweep(lat, s, beta, r), s)[1]
    arms = {"metropolis": sweep, "KBD": kbd, "sweep+KBD": lambda s, r: kbd(sweep(s, r), r)}
    for name, step in arms.items():
        res = []
        for seed in range(2):
            r = np.random.default_rng(seed)
            s = r.integers(2, size=lat.N)
            for _ in range(2000):
                s = step(s, r)
            X, t0 = np.empty((n, 3)), time.perf_counter()
            for t in range(n):
                s = step(s, r)
                f = np.fft.fft2((2 * s - 1).reshape(L, L))
                X[t] = abs(f[0, 0]) ** 2 / lat.N, abs(f[0, L // 2]) ** 2 / lat.N, lat.energy(s) / lat.N
            ms = 1e3 * (time.perf_counter() - t0) / n
            res.append([tau_int(X[:, k]) for k in range(3)] + [ms] + list(X.mean(0)))
        a = np.array(res).mean(0)
        print(f"T={T} {name:10s} tau S00={a[0]:7.1f} S0pi={a[1]:7.1f} E={a[2]:7.1f}  ms={a[3]:5.2f}  "
              f"max-tau*ms={max(a[:3])*a[3]:8.1f}  <S00>={a[4]:.2f} <S0pi>={a[5]:.2f} <E>/N={a[6]:.4f}", flush=True)
