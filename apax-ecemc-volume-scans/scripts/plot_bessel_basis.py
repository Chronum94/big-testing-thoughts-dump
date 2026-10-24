"""apax BesselBasis (Kocer form) with and without the cosine cutoff, n_basis 12, r_max 6 A: values and d/dr."""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E402,E702
from apax.layers.descriptor.basis_functions import BesselBasis, cosine_cutoff  # noqa: E402

N, RC = 12, 6.0
basis = BesselBasis(n_basis=N, r_max=RC, dtype="fp64")
d = jnp.linspace(0.0, RC, 1500)
B = lambda x: basis.apply({}, x)  # noqa: E731
BC = lambda x: basis.apply({}, x) * cosine_cutoff(x, RC)[:, None]  # noqa: E731
fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
for j, (f, lab) in enumerate(((B, "Bessel basis f_n(r)"), (BC, "f_n(r) × cosine cutoff"))):
    v, dv = jax.jvp(f, (d,), (jnp.ones_like(d),))
    for n in range(N):
        col = plt.cm.viridis(n / (N - 1))
        ax[0, j].plot(d, v[:, n], color=col, lw=1.2, label=f"n={n}" if j == 0 else None)
        ax[1, j].plot(d, dv[:, n], color=col, lw=1.2)
    ax[0, j].set_title(lab); ax[1, j].set_title(f"d/dr of {lab}")
    if j == 1:
        ax[0, j].plot(d, cosine_cutoff(d, RC) * float(jnp.abs(v).max()), "k--", lw=1, label="cutoff (scaled)")
    for a in ax[:, j]:
        a.axhline(0, color="k", lw=0.5); a.axvspan(0, 0.8, color="#eeeeee", zorder=0)
    ax[1, j].set_xlabel("r (Å)")
ax[0, 0].legend(fontsize=7, ncol=2); ax[0, 1].legend(fontsize=8)
ax[0, 0].text(0.05, 0.02, "grey: r < 0.8 Å, no data", transform=ax[0, 0].transAxes, fontsize=8, color="#666")
fig.suptitle(f"apax BesselBasis (Kocer), n_basis={N}, r_max={RC} Å: f_n = a(-1)^n [sinc((n+1)r/rc) + sinc((n+2)r/rc)]")
fig.tight_layout(); fig.savefig("bessel_basis.png", dpi=110); print("wrote bessel_basis.png")
