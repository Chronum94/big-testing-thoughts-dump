"""Error vs stored numbers per (pair, channel): cubic Hermite tables vs Chebyshev expansions (values from
interp_convergence.py and cheb_convergence.py, model arm_g20_r6t5_l1e3_s1, worst of 36 pairs x 17 channels)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E702
import numpy as np
hN = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192])
hR = [3.17e-2, 6.40e-3, 1.96e-3, 3.65e-4, 1.15e-4, 2.25e-5, 7.07e-6, 1.38e-6, 4.35e-7, 8.56e-8, 2.70e-8, 5.32e-9, 1.68e-9, 1.05e-10, 6.55e-12, 4.09e-13]
hdR = [5.16e-2, 1.70e-2, 7.20e-3, 2.09e-3, 8.65e-4, 2.53e-4, 1.07e-4, 3.13e-5, 1.32e-5, 3.89e-6, 1.64e-6, 4.84e-7, 2.04e-7, 2.55e-8, 3.18e-9, 3.98e-10]
sR = [2.64e-1, 5.18e-2, 1.95e-2, 4.05e-3, 1.26e-3, 2.39e-4, 7.32e-5, 1.39e-5, 4.30e-6, 8.29e-7, 2.59e-7, 5.04e-8, 1.59e-8, 9.71e-10, 6.04e-11, 3.76e-12]
sdR = [7.98e-1, 2.48e-1, 1.34e-1, 4.50e-2, 1.93e-2, 5.68e-3, 2.36e-3, 6.85e-4, 2.85e-4, 8.33e-5, 3.49e-5, 1.02e-5, 4.30e-6, 5.34e-7, 6.65e-8, 8.30e-9]
cN = np.array([8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96])
cR = [1.20e0, 4.42e-1, 8.88e-2, 1.45e-2, 1.02e-3, 2.93e-5, 4.27e-7, 1.82e-11, 1.38e-14, 1.01e-14, 9.32e-15, 1.15e-14, 1.60e-14]
cdR = [2.88e0, 1.85e0, 1.52e0, 2.26e-1, 2.18e-2, 1.39e-3, 3.13e-5, 2.22e-9, 1.56e-12, 1.41e-12, 2.68e-12, 1.74e-12, 3.32e-12]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, (h, s, c), lab in ((ax[0], (hR, sR, cR), "R"), (ax[1], (hdR, sdR, cdR), "dR/dr (forces)")):
    a.loglog(2 * hN, h, "o-", color="#2a6fdb", label="cubic Hermite (value + derivative per node)")
    a.loglog(hN, s, "^-", color="#e0672b", label="cubic spline (value per node)")
    a.loglog(cN, c, "s-", color="#2ca02c", label="Chebyshev expansion (coefficients)")
    a.axvline(12, color="#7f7f7f", ls="--", lw=1); a.text(12.5, 1e-13, "model's own\n12 Bessel coeffs\n(exact)", fontsize=8, color="#555")
    a.axhline(1.2e-7, color="k", ls=":", lw=0.8); a.text(9, 2e-7, "fp32 epsilon", fontsize=8)
    a.set(xlabel="stored numbers per (element pair, channel)", ylabel=f"max relative error in {lab}", title=lab, ylim=(1e-15, 5))
    a.legend(fontsize=8, loc="lower left")
fig.suptitle("Approximating the learned radials (36 element pairs × 17 channels, worst case): Hermite vs spline vs Chebyshev")
fig.tight_layout(); fig.savefig("interp_vs_cheb.png", dpi=110); print("wrote interp_vs_cheb.png")
