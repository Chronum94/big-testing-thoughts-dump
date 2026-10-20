"""Arrhenius plot of Li self-diffusion from student MD vs experiment (Nernst-Einstein, Haven ratio 1)."""
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

kB, e, kJ = 8.617333e-5, 1.602177e-19, 1.380649e-23


def D_of(d, ps=100.0):
    """Li self-diffusion (cm^2/s): MSD over time origins after the first 10 %, lags up to 20 % of the run."""
    f = h5py.File(f"md/{d}/md.h5", "r"); p = f["particles/atoms"]; n = len(p["position/time"])
    Z = p["species/value"][0].astype(int); box = p["box/edges/value"][0]; li = Z == 3
    fr = p["position/value"][:n][:, li] @ np.linalg.inv(box); dfr = np.diff(fr, axis=0); dfr -= np.round(dfr)
    un = np.concatenate([fr[:1], fr[:1] + np.cumsum(dfr, axis=0)]) @ box
    t = np.linspace(0, ps, n); lag, s = int(n * 0.2), n // 10
    msd = np.array([((un[s + L:] - un[s:-L]) ** 2).sum(-1).mean() for L in range(1, lag)]); tl = t[1:lag]; k = len(tl) // 4
    return np.polyfit(tl[k:], msd[k:], 1)[0] / 6 * 1e-4


def D_from_sigma(sig_mS, T, n_li=20, V=960.8):  # experimental cell (Kuhn 2013, 298 K)
    n = n_li / V * 1e24  # cm^-3
    return sig_mS * 1e-3 * kJ * T / (n * e**2)


T = np.array([600, 700, 800, 900]); D = np.array([D_of(f"lgB5_s1_dt2_nvt{t}") for t in T])
T05 = np.array([600, 900]); D05 = np.array([D_of(f"lgB5_s1_nvt{t}") for t in T05])
sl, ic = np.polyfit(1 / (kB * T), np.log(D), 1); Ea = -sl
x = np.linspace(1000 / 950, 1000 / 290, 100)
fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
ax.plot(x, np.exp(sl * x / 1000 / kB + ic), c="#1f5fa8", lw=1.2, ls="--", label=f"fit, Ea = {Ea:.3f} eV (extrapolated below 600 K)")
ax.scatter(1000 / T, D, c="#1f5fa8", s=55, zorder=3, edgecolors="white", linewidths=1.2, label="apax student, NVT, dt 2 fs, 100 ps")
ax.scatter(1000 / T05, D05, facecolors="none", edgecolors="#1f5fa8", s=55, zorder=3, label="same, dt 0.5 fs")
for sig, Tk, lab, mk in [(9, 298, "exp. 9 mS/cm, Kuhn 2013 (EES), Ea 0.21–0.22 eV", "s"), (12, 300, "exp. 12 mS/cm total, pellet, Kamaya 2011", "D")]:
    d0 = D_from_sigma(sig, Tk)
    ax.scatter(1000 / Tk, d0, marker=mk, c="#d1642b", s=60, zorder=3, label=lab)
    if sig == 9:
        xs = np.linspace(1000 / 950, 1000 / Tk, 50)
        ax.plot(xs, d0 * np.exp(-0.215 / kB * (xs / 1000 - 1 / Tk)), c="#d1642b", lw=1, alpha=0.6, label="exp. point with Ea = 0.215 eV")
ax.set_yscale("log"); ax.set_xlabel("1000 / T  (K⁻¹)"); ax.set_ylabel("Li self-diffusion D (cm²/s)")
sec = ax.secondary_xaxis("top", functions=(lambda v: 1000 / v, lambda v: 1000 / v)); sec.set_xlabel("T (K)"); sec.set_xticks([900, 800, 700, 600, 500, 400, 300])
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("Li₁₀GeP₂S₁₂: student MD (820-frame bootstrap) vs experiment\nexperiment converted from σ by Nernst–Einstein, Haven ratio 1; MD at fixed 0 K volume", fontsize=10)
fig.savefig("arrhenius.png", dpi=130)
print("Ea", Ea, "D", D, "D0.5fs", D05)
