"""Plot a trained model's radial functions R_r(d) (all n_radial channels, feeding the scalar moment) and the
compressed tensor channels (radial_compression applied, feeding the l = 1-3 moments) vs distance, per element pair.

usage: python plot_radials.py MODEL_DIR OUT.png   (CPU)
Pairs are (centre, neighbour); the radial includes the cosine cutoff, as the descriptor sees it.
"""
import sys

sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401
import apax.train.checkpoints as ck  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E402,E702
import numpy as np  # noqa: E402
from apax.nn.builder import ModelBuilder  # noqa: E402

cfg, params = ck.restore_parameters(sys.argv[1])
rep = params["params"]["energy_model"]["representation"]
radial = ModelBuilder(cfg.model.model_dump()).build_radial_function()
W = np.asarray(rep["radial_compression"]["kernel"])
Z = {"H": 1, "Li": 3, "C": 6, "O": 8, "F": 9, "P": 15}
PAIRS = [("Li", "O"), ("O", "Li"), ("Li", "F"), ("C", "O"), ("O", "H"), ("P", "F")]
d = jnp.linspace(0.5, radial.basis_fn.r_max, 600)
fig, ax = plt.subplots(2, len(PAIRS), figsize=(3.4 * len(PAIRS), 6.4), sharex=True)
for j, (c, n) in enumerate(PAIRS):
    zc, zn = jnp.full(d.shape, Z[c]), jnp.full(d.shape, Z[n])
    R = np.asarray(radial.apply({"params": rep["radial_fn"]}, d, zn, zc))  # call convention (dr, Z_neighbour, Z_centre)
    for k, (Y, cm) in enumerate(((R, plt.cm.viridis), (R @ W, plt.cm.plasma))):
        for r in range(Y.shape[1]):
            ax[k, j].plot(d, Y[:, r], color=cm(r / max(Y.shape[1] - 1, 1)), lw=1.1)
        ax[k, j].axhline(0, color="k", lw=0.5)
    ax[0, j].set_title(f"centre {c} ← neighbour {n}")
    ax[1, j].set_xlabel("distance (Å)")
ax[0, 0].set_ylabel(f"radial channels ({R.shape[1]})"); ax[1, 0].set_ylabel(f"tensor channels ({W.shape[1]})")
fig.suptitle(f"{sys.argv[1].rstrip('/').split('/')[-1]}: learned radial functions (incl. cosine cutoff); colour = channel index")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110); print("wrote", sys.argv[2])
