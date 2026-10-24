"""Table size vs arithmetic per pair for radial evaluation backends, at two force-accuracy targets (max relative
dR/dr error, worst of 36 element pairs; errors from interp_convergence.py, cheb_convergence.py, piecewise_convergence.py,
piecewise_check.py on arm_g20_r6t5_l1e3_s1). Costs per pair for the 12 radial channels (compression is separate):
exact pair table 12 x 12 MAC (+24 sin as written, +2 sin and 12 FMA with the sine recurrence); cubic Hermite 4 x 12;
global Chebyshev n recurrence + n x 12; piecewise degree p: (p + 1) x 12 Horner. Table = 36 pairs x 12 channels, fp32."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E702
import numpy as np

KB = 36 * 12 * 4 / 1024  # one stored number per (pair, channel) across the table, in KB
pts = []  # (label, family, table KB, FMA per pair, dR error)
for N, e in zip([16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192],
                [5.16e-2, 1.70e-2, 7.20e-3, 2.09e-3, 8.65e-4, 2.53e-4, 1.07e-4, 3.13e-5, 1.32e-5, 3.89e-6, 1.64e-6, 4.84e-7, 2.04e-7, 2.55e-8, 3.18e-9, 3.98e-10]):
    pts.append((f"N{N}", "cubic Hermite", 2 * N * KB, 4 * 12 + 10, e))
for n, e in zip([8, 12, 16, 20, 24, 28, 32, 40, 48], [2.88, 1.85, 1.52, 0.226, 2.18e-2, 1.39e-3, 3.13e-5, 2.22e-9, 1.56e-12]):
    pts.append((f"n{n}", "global Chebyshev", n * KB, 13 * n, e))
for (p, K), e in {(5, 16): 1.3e-3, (5, 32): 5.4e-5, (5, 64): 1.8e-6, (7, 8): 2.9e-3, (7, 16): 2.9e-5, (7, 32): 2.6e-7,
                  (9, 8): 1.6e-4, (9, 16): 4.3e-7}.items():
    pts.append((f"{p}×{K}", "piecewise Hermite (C^m smooth)", K * (p + 1) * KB, (p + 1) * 12, e))
for (p, K), e in {(3, 32): 8.6e-3, (3, 64): 1.1e-3, (3, 128): 1.3e-4, (5, 8): 2.5e-2, (5, 16): 9.4e-4, (5, 32): 3.6e-5,
                  (5, 64): 1.2e-6, (5, 128): 3.6e-8, (7, 8): 8.8e-4, (7, 16): 8.1e-6, (7, 32): 6.9e-8, (9, 8): 1.8e-5,
                  (9, 16): 4.3e-8}.items():
    pts.append((f"{p}×{K}", "piecewise Chebyshev pts (not smooth)", K * (p + 1) * KB, (p + 1) * 12, e))
pts.append(("exact", "exact pair table (no error)", 12 * KB, 144, 0.0))
pts.append(("exact+rec", "exact pair table + sine recurrence", 12 * KB, 156, 0.0))
STYLE = {"cubic Hermite": ("o", "#2a6fdb"), "global Chebyshev": ("s", "#2ca02c"),
         "piecewise Hermite (C^m smooth)": ("D", "#d62728"), "piecewise Chebyshev pts (not smooth)": ("d", "#ff9896"),
         "exact pair table (no error)": ("*", "#000000"), "exact pair table + sine recurrence": ("P", "#7f7f7f")}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
for a, tol in zip(ax, (1e-4, 1e-6)):
    ok = [q for q in pts if q[4] <= tol]
    # cheapest config per family meeting the target, plus every family member for context
    for fam, (mk, col) in STYLE.items():
        fam_pts = [q for q in ok if q[1] == fam]
        if not fam_pts:
            continue
        a.scatter([q[2] for q in fam_pts], [q[3] for q in fam_pts], marker=mk, color=col, s=70 if mk != "*" else 220,
                  label=fam, zorder=3, edgecolor="k" if mk in "*P" else None)
        best = min(fam_pts, key=lambda q: q[2])
        a.annotate(f"{best[0]}\n(dR {best[4]:.0e})" if best[4] else best[0], (best[2], best[3]), xytext=(6, 4),
                   textcoords="offset points", fontsize=7.5)
    front = sorted(ok, key=lambda q: (q[2], q[3])); f_x, f_y, ymin = [], [], np.inf
    for q in front:
        if q[3] < ymin:
            f_x.append(q[2]); f_y.append(q[3]); ymin = q[3]
    a.step(f_x, f_y, where="post", color="k", lw=1, ls="--", alpha=0.6, label="Pareto front", zorder=2)
    for x, lab in ((32, "L1 32 KB"), (64, "LDS 64 KB"), (8192, "L2 8 MB")):
        a.axvline(x, color="#bbbbbb", lw=1); a.text(x * 1.05, 600, lab, fontsize=8, color="#777", rotation=90, va="top")
    a.set(xscale="log", yscale="log", xlabel="table size, all 36 element pairs (KB, fp32)",
          title=f"configs with max relative dR/dr error ≤ {tol:.0e}")
ax[0].set_ylabel("FMA per pair (12 radial channels)"); ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Radial evaluation: table size vs arithmetic (per-centre-element blocking divides the table by 6)")
fig.tight_layout(); fig.savefig("table_pareto.png", dpi=110); print("wrote table_pareto.png")
