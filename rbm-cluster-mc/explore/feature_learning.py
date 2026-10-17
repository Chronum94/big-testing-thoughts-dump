"""Learn which Wang-BM features to use (FeatureBM, 24 params) by maximising the expected squared jump
(ESJD) of Z2-invariant slow observables per composite step (Metropolis sweep + feature move).
FFIM L=8, T=0.5. Antithetic evolution strategy with common random numbers. Saves theta to .npy.
usage: feature_learning.py <init: off|random> <seed>"""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from bm3 import FeatureBM

init, seed = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("off", 0)
L, T = 8, 0.5
beta, lat = 1 / T, Lattice(L, 2, q=2, kind="ffim")
fbm = FeatureBM(lat)
rng = np.random.default_rng(seed)


def obs(s):
    f = np.fft.fft2((2 * s - 1).reshape(L, L))
    return np.array([abs(f[0, 0]) ** 2, abs(f[0, L // 2]) ** 2, lat.energy(s)]) / lat.N


# pilot Metropolis run: equilibrated start states + observable scales
s, starts, O = rng.integers(2, size=lat.N), [], []
for t in range(6000):
    metropolis_sweep(lat, s, beta, rng)
    if t >= 1000 and t % 50 == 0:
        starts.append(s.copy())
    if t >= 1000:
        O.append(obs(s))
scale = np.array(O).std(0)
starts = starts[:16]


def esjd(theta, crn_seed, k=20):
    """Mean over starts and steps of |delta obs / scale|^2 per composite step. Same seed => same randomness."""
    total = 0.0
    for j, s0 in enumerate(starts):
        r = np.random.default_rng([crn_seed, j])
        s, o = s0.copy(), obs(s0)
        for _ in range(k):
            metropolis_sweep(lat, s, beta, r)
            s = fbm.step(lat, beta, fbm.typed(theta), s, r)[0]
            o2 = obs(s)
            total += (((o2 - o) / scale) ** 2).sum()
            o = o2
    return total / (len(starts) * k)


n_par = 2 * FeatureBM.N_TYPES
theta = (np.concatenate([np.zeros(12), np.full(12, -4.0)]) if init == "off"
         else np.concatenate([rng.normal(0, 1, 12), rng.normal(-2, 1, 12)]))
kbd = None
print(f"reference ESJD: all-off={esjd(np.concatenate([np.zeros(12), np.full(12, -30.)]), 999):.3f}  (KBD: see feature_learning_exact.py)", flush=True)
sigma, lr, n_dir = 0.7, 0.4, 8
for it in range(40):
    eps = rng.normal(size=(n_dir, n_par))
    f = np.array([[esjd(theta + sgn * sigma * e, 1000 + it) for sgn in (1, -1)] for e in eps])
    diff = f[:, 0] - f[:, 1]
    theta = theta + lr * (diff / (np.abs(diff).mean() + 1e-12)) @ eps / n_dir   # normalised antithetic step
    theta = np.clip(theta, -12, 12)
    if it % 5 == 4 or it == 0:
        print(f"it {it+1:3d}  ESJD(theta)={esjd(theta, 999):.3f}  mean f={f.mean():.3f}", flush=True)
np.save(f"explore/theta_{init}_{seed}.npy", theta)

names = [f"{k}/sub{sb}/{'frus' if fr else 'unfr'}" for k in ("pair0", "pair1", "plaq") for sb in (0, 1) for fr in (0, 1)]
print("\nlearned activations p(h=1|F) for frustrated types (FFIM has only frustrated plaquettes):")
for t, nm in enumerate(names):
    if "frus" in nm:
        W, b = theta[t], theta[12 + t]
        Fs = (-2, 2) if nm.startswith("plaq") else (-2, 0, 2)
        print(f"  {nm:18s} W={W:6.2f} b={b:6.2f}  " + "  ".join(f"p(F={F:+d})={1/(1+np.exp(-(W*F+b))):.3f}" for F in Fs))
