"""Tight-SCF row of rootfind_sim, anchored methods vs the current ones."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rootfind_sim as rs

KEEP = [('master (regress + mix)', '#2a78d6'),
        ('branch (weighted, replace guess)', '#eb6834'),
        ('anchored linear', '#1baf7a'),
        ('anchored inverse-quadratic', '#eda100')]
methods = dict(rs.METHODS)
fig, axes = plt.subplots(1, len(rs.CASES), figsize=(16, 3.8), sharey=True,
                         facecolor='#fcfcfb')
for ax, (cname, phi_of, n0, target) in zip(axes, rs.CASES):
    for mname, col in KEEP:
        rng = np.random.default_rng(0)
        out = [rs.run(methods[mname], phi_of, n0, target, False, rng)
               for _ in range(rs.NSEED)]
        L = max(len(o[1]) for o in out)
        E = np.full((len(out), L), rs.TOL / 10)
        for i, o in enumerate(out):
            E[i, :len(o[1])] = o[1]
        med = np.median(np.log10(np.maximum(E, 1e-4)), axis=0)
        ax.plot(np.arange(1, L + 1), med, '-o', color=col, lw=2, ms=4,
                label=mname)
    ax.axhline(np.log10(rs.TOL), color='#6b6a64', ls='--', lw=1)
    ax.set_title(cname, loc='left', fontsize=9)
    ax.set_xlabel('phi evaluation (SCF)')
    ax.set_xlim(0.5, 19.5)
    ax.set_facecolor('#fcfcfb')
    ax.grid(color='#e6e5e0', lw=0.8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('tight SCF: median log10 |phi - phi*|')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig('rootfind_anchored_tight.png', dpi=120, facecolor='#fcfcfb')
