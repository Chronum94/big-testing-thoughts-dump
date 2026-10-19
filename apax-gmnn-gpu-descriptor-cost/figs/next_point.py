"""GP on log val-F over (log2 A, log2 B, l_max) + linear MD-cost model -> expected improvement of
the accuracy/MD-time Pareto front for candidate configs. molcrys, 400 epochs, seed 1."""
import csv, json, os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"v2B4_s1": (12, 4, 3), "p1_B2": (12, 2, 3), "p1_B3": (12, 3, 3), "p1_B6": (12, 6, 3),
        "p1_B8": (12, 8, 3), "p1_B5": (12, 5, 3), "p1_L3B10": (12, 10, 2), "p1_L3B8": (12, 8, 2), "p1_A16B5": (16, 5, 3), "v2L3_s1": (12, 4, 2), "p1_L3A24": (24, 4, 2), "p1_L3A32": (32, 4, 2),
        "p1_L3B6": (12, 6, 2)}
cost = [json.loads(l) for l in open(f"{ROOT}/costmap/costmap.jsonl")]

# cost model: MD ms ~ a + b*floats_per_pair + c*B^2*(l-dependent), fit per l_max on the measured points
def cfeat(A, B, l):
    fpp = A + (39 if l == 3 else 12) * B
    return [1, fpp, B**2]
cm = [(c["A"], c["B"], 3 if 3 not in c["drop"] else 2, c["md_ms_box12960"]) for c in cost
      if c["drop"] in ([], [3, 6, 7])]
X = np.array([cfeat(a, b, l) for a, b, l, _ in cm]); y = np.array([m for *_, m in cm])
coef = np.linalg.lstsq(X, y, rcond=None)[0]
md_meas = {}
for a, b, l, m in cm:
    md_meas.setdefault((a, b, l), []).append(m)
md_meas = {k: float(np.mean(v)) for k, v in md_meas.items()}  # repeats averaged
md = lambda A, B, l: md_meas.get((A, B, l), float(np.dot(cfeat(A, B, l), coef)))  # measured: cost is a staircase in B (B5 -> B6 +52 %)
res = y - X @ coef
print(f"cost model rms resid {np.sqrt(np.mean(res**2)):.2f} ms over {len(y)} measured configs")

obs = []
for r, (A, B, l) in RUNS.items():
    if not os.path.exists(f"{ROOT}/logs/{r}.csv"):
        continue
    rows = list(csv.DictReader(open(f"{ROOT}/logs/{r}.csv")))
    if len(rows) >= 400:
        obs.append((A, B, l, 1000 * float(rows[-1]["val_forces_mae"])))
feat = lambda A, B, l: [np.log2(A), np.log2(B), float(l == 3)]
Xo = np.array([feat(*o[:3]) for o in obs]); yo = np.log(np.array([o[3] for o in obs]))
k = C(1, (1e-2, 1e2)) * RBF([2, 2, 1], (0.5, 20)) + WhiteKernel(1e-5, (1e-7, 1e-3))
gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=20, random_state=0).fit(Xo, yo)
print(f"{len(obs)} observations; kernel: {gp.kernel_}")

front = sorted((md(*o[:3]), o[3]) for o in obs)
def best_at(c):  # best observed F at MD cost <= c
    v = [f for m, f in front if m <= c]
    return min(v) if v else np.inf

seen = {o[:3] for o in obs}
BUDGET = 31.0  # ms: 2x L0; plain EI otherwise chases the most expensive B
cand = [(A, B, l) for A in (8, 12, 16, 24, 32) for B in (2, 3, 4, 5, 6, 8, 10, 12) for l in (2, 3)
        if (A, B, l) not in seen and md(A, B, l) <= BUDGET]
mu, sd = gp.predict(np.array([feat(*c) for c in cand]), return_std=True)
out = []
for (A, B, l), m, s in zip(cand, mu, sd):
    c = md(A, B, l); f0 = best_at(c)
    if not np.isfinite(f0):
        f0 = front[0][1]  # cheaper than everything observed: must beat the cheapest observed point
    z = (np.log(f0) - m) / s
    ei = (np.log(f0) - m) * norm.cdf(z) + s * norm.pdf(z)  # EI in log F (fractional improvement)
    meas = "meas" if (A, B, l) in md_meas else "est"
    out.append((ei, A, B, l, c, meas, np.exp(m), np.exp(m) * s, f0, norm.cdf(z)))
out.sort(reverse=True)
print(f"{'A':>3} {'B':>3} {'l':>2} {'MDms':>5}      {'predF':>6} {'±':>4} {'frontF':>6} {'P(beat)':>7} {'EI %':>5}")
for ei, A, B, l, c, meas, f, e, f0, p in out[:10] + [o for o in out if o[1:4] == (12, 8, 2)]:
    print(f"{A:3} {B:3} {l:2} {c:5.1f} {meas:4} {f:6.1f} {e:4.1f} {f0:6.1f} {p:7.2f} {100*ei:5.2f}")
