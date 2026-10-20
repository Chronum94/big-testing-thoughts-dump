"""Directional Li diffusion (c vs ab) from the dt 2 fs NVT sweep: Arrhenius fits and plot."""
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

kB = 8.617333e-5


def D_dir(d, ps=100.0):
    """(D_ab, D_c, D_total) in cm^2/s; c = third cell vector (z), MSD over time origins after the first 10 %."""
    f = h5py.File(f"md/{d}/md.h5", "r"); p = f["particles/atoms"]; n = len(p["position/time"])
    Z = p["species/value"][0].astype(int); box = p["box/edges/value"][0]; li = Z == 3
    fr = p["position/value"][:n][:, li] @ np.linalg.inv(box); dfr = np.diff(fr, axis=0); dfr -= np.round(dfr)
    un = np.concatenate([fr[:1], fr[:1] + np.cumsum(dfr, axis=0)]) @ box
    t = np.linspace(0, ps, n); lag, s = int(n * 0.2), n // 10
    msd = np.array([((un[s + L:] - un[s:-L]) ** 2).mean((0, 1)) for L in range(1, lag)])  # (lag, 3) per axis
    tl = t[1:lag]; k = len(tl) // 4
    sl = [np.polyfit(tl[k:], msd[k:, a], 1)[0] for a in range(3)]  # A^2/ps per axis = 2 D_axis
    return (sl[0] + sl[1]) / 4 * 1e-4, sl[2] / 2 * 1e-4, sum(sl) / 6 * 1e-4


T = np.array([600, 700, 800, 900]); D = np.array([D_dir(f"lgB5_s1_dt2_nvt{t}") for t in T])
print("cell angles:", np.round(h5py.File("md/lgB5_s1_dt2_nvt600/md.h5")["particles/atoms/box/edges/value"][0], 3).tolist())
fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
x = np.linspace(1000 / 950, 1000 / 290, 100); fits = {}
for col, lab, c in [(1, "c axis", "#1f5fa8"), (0, "ab plane", "#d1642b"), (2, "total", "#777")]:
    sl, ic = np.polyfit(1 / (kB * T), np.log(D[:, col]), 1); fits[lab] = (-sl, np.exp(sl / (kB * 300) + ic))
    ax.plot(x, np.exp(sl * x / 1000 / kB + ic), c=c, lw=1, ls="--")
    ax.scatter(1000 / T, D[:, col], c=c, s=50, zorder=3, edgecolors="white", label=f"{lab}: Ea {-sl:.3f} eV")
for t, row in zip(T, D):
    print(f"{t} K: D_c {row[1]:.2e}  D_ab {row[0]:.2e}  D_tot {row[2]:.2e}  c/ab {row[1] / row[0]:.2f}")
for k, (ea, d300) in fits.items():
    print(f"{k:8s}: Ea {ea:.3f} eV, D(300 K) extrapolated {d300:.2e} cm2/s")
print(f"c/ab at 300 K (extrapolated): {fits['c axis'][1] / fits['ab plane'][1]:.1f}")
ax.set_yscale("log"); ax.set_xlabel("1000 / T  (K⁻¹)"); ax.set_ylabel("Li self-diffusion D (cm²/s)")
sec = ax.secondary_xaxis("top", functions=(lambda v: 1000 / np.maximum(v, 1e-9), lambda v: 1000 / np.maximum(v, 1e-9)))
sec.set_xlabel("T (K)"); sec.set_xticks([900, 800, 700, 600, 500, 400, 300])
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9, frameon=False, loc="lower left")
ax.set_title("LGPS Li diffusion by direction (student MD, NVT, fixed 0 K volume)\n"
             "references: PBE AIMD (Mo 2012) Ea 0.17 eV (c) / 0.28 eV (ab); single crystal (Iwasaki 2019) 27 vs 7 mS/cm at RT", fontsize=9)
fig.savefig("arrhenius_aniso.png", dpi=130)
