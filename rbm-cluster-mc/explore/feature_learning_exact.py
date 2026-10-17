"""Structure learning inside the rejection-free family FeatureBM.exact_split: how should each bond's coupling
be split between the sublattice-0 plaquette, the sublattice-1 plaquette and its own link feature, and with
what biases? 6 params: logits (a0, a1, a_link) -> softmax shares, and b0, b1, b_link.
KBD = (1, 0, 0) with b0 -> -inf; SW = (0, 0, 1) with b_link -> -inf.
Objective as feature_learning.py: ESJD of Z2-invariant slow observables per (Metropolis sweep + move), FFIM L=8 T=0.5.
usage: feature_learning_exact.py <init: sw|split|random> <seed>"""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from bm3 import FeatureBM, OFF

init, seed = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("split", 0)
L, T = 8, 0.5
beta, lat = 1 / T, Lattice(L, 2, q=2, kind="ffim")
fbm = FeatureBM(lat)
rng = np.random.default_rng(seed)


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


def prm_of(x):
    a = np.exp(x[:3] - x[:3].max()); a /= a.sum()
    b0, b1, bl = np.clip(x[3:], OFF, 8)
    return fbm.exact_split(lat, beta, a[0], a[1], b0, b1, bl), a


def esjd(prm, crn_seed, k=20):
    total, acc = 0.0, 0
    for j, s0 in enumerate(starts):
        r = np.random.default_rng([crn_seed, j])
        s, o = s0.copy(), obs(s0)
        for _ in range(k):
            metropolis_sweep(lat, s, beta, r)
            s, a, _ = fbm.step(lat, beta, prm, s, r)
            acc += a
            o2 = obs(s)
            total += (((o2 - o) / scale) ** 2).sum()
            o = o2
    return total / (len(starts) * k), acc / (len(starts) * k)


refs = {"KBD (1,0,0)": fbm.kbd(lat, beta),
        "SW (0,0,1)": fbm.exact_split(lat, beta, 0, 0, OFF, OFF, OFF),
        "split (.5,.5,0)": fbm.exact_split(lat, beta, .5, .5, OFF, OFF, OFF),
        "metropolis only": dict(W=np.zeros((lat.N, 3)), b=np.full((lat.N, 3), OFF), Wl=np.zeros(len(lat.J)), bl=np.full(len(lat.J), OFF))}
for nm, prm in refs.items():
    e, a = esjd(prm, 999)
    print(f"reference {nm:16s} ESJD={e:.3f} acc={a:.3f}", flush=True)

x = {"sw": np.array([-5, -5, 5, -3, -3, -10.]), "split": np.array([0, 0, 0, 0, 0, 0.]),
     "random": np.concatenate([rng.normal(0, 2, 3), rng.normal(-2, 3, 3)])}[init]
sigma, lr, n_dir = 0.5, 0.3, 8
for it in range(30):
    eps = rng.normal(size=(n_dir, 6))
    f = np.array([[esjd(prm_of(x + sg * sigma * e)[0], 1000 + it)[0] for sg in (1, -1)] for e in eps])
    diff = f[:, 0] - f[:, 1]
    x = x + lr * (diff / (np.abs(diff).mean() + 1e-12)) @ eps / n_dir
    x[3:] = np.clip(x[3:], -15, 8)
    if it % 5 == 4 or it == 0:
        prm, a = prm_of(x)
        e, ac = esjd(prm, 999)
        print(f"it {it+1:3d} shares(sub0,sub1,link)=({a[0]:.2f},{a[1]:.2f},{a[2]:.2f}) b=({x[3]:.1f},{x[4]:.1f},{x[5]:.1f}) "
              f"ESJD={e:.3f} acc={ac:.3f}", flush=True)
np.save(f"explore/xexact_{init}_{seed}.npy", x)
