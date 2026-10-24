"""Reset vs kept mixer history between N updates (slope-guess runs)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_convergence import parse, SURFACE, INK, MUTED, GRID  # noqa

SYSTEMS = [('Au40', 'Au(111), 4.0 V'), ('Pt45', 'Pt(111), 4.5 V'),
           ('Cu40', 'Cu(111), 4.0 V')]
RUNS = [('est', 'mixer reset at each N update (current)', '#2a78d6'),
        ('keep_est', 'mixer history kept', '#eb6834')]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex='col',
                         facecolor=SURFACE)
for j, (key, title) in enumerate(SYSTEMS):
    for tag, label, color in RUNS:
        dens, wf, att = parse(f'{tag}_{key}.txt')
        it = np.arange(1, len(dens) + 1)
        ends = np.array([a[0] for a in att]) - 1
        energy = [float(l.split()[3].rstrip('c'))
                  for l in open(f'{tag}_{key}.txt') if l.startswith('iter:')]
        for row, y in [(0, dens), (1, np.array(energy))]:
            axes[row, j].plot(it, y, color=color, lw=1.3, label=label)
            axes[row, j].plot(it[ends], y[ends], 'o', color=color, ms=5,
                              mec=SURFACE, mew=1)
    lo, hi = np.inf, -np.inf
    for tag, _, _ in RUNS:
        e = [float(l.split()[3].rstrip('c'))
             for l in open(f'{tag}_{key}.txt') if l.startswith('iter:')][10:]
        lo, hi = min(lo, min(e)), max(hi, max(e))
    axes[1, j].set_ylim(lo - 0.3, hi + 0.3)
    axes[0, j].axhline(-4, color=MUTED, ls='--', lw=1)
    axes[0, j].set_title(title, loc='left', fontsize=10, color=INK)
    axes[1, j].set_xlabel('cumulative SCF iterations')
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
axes[0, 0].set_ylabel('log10 density residual')
axes[1, 0].set_ylabel('SCF energy (eV)')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('keep_mixer.png', dpi=130, facecolor=SURFACE)
