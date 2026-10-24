"""Chebyshev expansion of the tabulated radials: max relative error vs number of coefficients (same reference and
error definition as interp_convergence.py). usage: python cheb_convergence.py MODEL_DIR [R_MIN]   (CPU, float64)
Coefficients from interpolation at Chebyshev points of the first kind (exact up to aliasing); derivative via chebder.
"""
import sys
import numpy as np
from numpy.polynomial import chebyshev as C
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null"] + sys.argv[2:]
src = open("/home/chronum/scratch/ecemc/interp_convergence.py").read().split("fine = np.linspace")[0]
exec(src)  # noqa: S102  (defines radial, rep, W, r_min, r_max, pairs, R_and_dR)
fine = np.linspace(r_min, r_max, 20001); x = (2 * fine - r_min - r_max) / (r_max - r_min)
ref = [R_and_dR(jnp.asarray(fine), c, n) for c, n in pairs]
for n in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96):
    k = np.arange(n); xn = np.cos(np.pi * (k + 0.5) / n); dn = 0.5 * (xn * (r_max - r_min) + r_min + r_max)
    eR = edR = 0.0
    for (c, m), (Rf, dRf) in zip(pairs, ref):
        Rn, _ = R_and_dR(jnp.asarray(dn), c, m)
        coef = C.chebfit(xn, Rn, n - 1)                      # (n, channels)
        P = C.chebval(x, coef).T; dP = C.chebval(x, C.chebder(coef)).T * 2 / (r_max - r_min)
        eR = max(eR, (np.abs(P - Rf) / (np.abs(Rf).max(0) + 1e-30)).max())
        edR = max(edR, (np.abs(dP - dRf) / (np.abs(dRf).max(0) + 1e-30)).max())
    print(f"coeffs {n:3d}  R {eR:.2e}  dR {edR:.2e}  table {36 * n * 17 * 4 / 1024:.0f} KB (fp32)", flush=True)
