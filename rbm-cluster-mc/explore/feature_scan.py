"""Coarse structure scan of the rejection-free family FeatureBM.exact_split on FFIM L=8, T=0.5:
ESJD of Z2-invariant slow observables per (sweep + move), 3 CRN seeds -> mean +- sem.
Also the fraction of moves where the largest frozen cluster spans > 90% of sites (percolation = useless global flip)."""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from bm3 import FeatureBM, OFF
from rbm import clusters

L, T = 8, 0.5
beta, lat = 1 / T, Lattice(L, 2, q=2, kind="ffim")
fbm, rng = FeatureBM(lat), np.random.default_rng(0)


def obs(s):
    f = np.fft.fft2((2 * s - 1).reshape(L, L))
    return np.array([abs(f[0, 0]) ** 2, abs(f[0, L // 2]) ** 2, lat.energy(s)]) / lat.N


s, starts, O = rng.integers(2, size=lat.N), [], []
for t in range(6000):
    metropolis_sweep(lat, s, beta, rng)
    if t >= 1000 and t % 50 == 0:
        starts.append(s.copy())
    if t >= 1000:
        O.append(obs(s))
scale, starts = np.array(O).std(0), starts[:16]


def esjd(prm, crn_seed, k=20):
    total, perc = 0.0, 0
    for j, s0 in enumerate(starts):
        r = np.random.default_rng([crn_seed, j])
        s, o = s0.copy(), obs(s0)
        for _ in range(k):
            metropolis_sweep(lat, s, beta, r)
            _, frozen = fbm.proposal(lat, prm, s, np.random.default_rng([crn_seed, j, 7]))
            perc += np.bincount(clusters(lat.N, lat.i[frozen], lat.j[frozen])).max() > 0.9 * lat.N
            s = fbm.step(lat, beta, prm, s, r)[0]
            o2 = obs(s)
            total += (((o2 - o) / scale) ** 2).sum()
            o = o2
    return total / (len(starts) * k), perc / (len(starts) * k)


points = [("metropolis only", None)]
points += [(f"a0={a0:.2f} a1=0 link={1-a0:.2f} b=off", (a0, 0, OFF, OFF, OFF)) for a0 in (0, .25, .5, .75, .9, 1)]
points += [(f"a0={a0:.2f} a1={1-a0:.2f} link=0 b=off", (a0, 1 - a0, OFF, OFF, OFF)) for a0 in (.5, .75)]
points += [(f"KBD a0=1, b0={b0:g}", (1, 0, b0, OFF, OFF)) for b0 in (-2, 0, 2)]
for name, p in points:
    prm = (dict(W=np.zeros((lat.N, 3)), b=np.full((lat.N, 3), OFF), Wl=np.zeros(len(lat.J)), bl=np.full(len(lat.J), OFF))
           if p is None else fbm.exact_split(lat, beta, *p))
    res = np.array([esjd(prm, c) for c in (1, 2, 3)])
    print(f"{name:32s} ESJD={res[:, 0].mean():.3f} +- {res[:, 0].std(ddof=1)/np.sqrt(3):.3f}  percolating={res[:, 1].mean():.2f}", flush=True)
