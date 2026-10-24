"""Outer (potential) and inner (SCF residual) convergence from SJM logs."""
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#6b6a64', '#e6e5e0'
RUNS = [('base', 'master', '#2a78d6'),
        ('est', '+ slope guess', '#eb6834'),
        ('inx1', '+ inexact Newton (0.1)', '#1baf7a')]
SYSTEMS = [('Au40', 'Au(111), 4.0 V', 4.0), ('Pt45', 'Pt(111), 4.5 V', 4.5),
           ('Cu40', 'Cu(111), 4.0 V', 4.0)]


def parse(fname):
    """Return per-iteration (dens, wkfxn) log10 residuals and, per attempt,
    (cumulative iterations, potential)."""
    dens, wf, attempts = [], [], []
    for line in open(fname):
        if line.startswith('iter:'):
            cols = [c.rstrip('c') for c in line.split()[4:]]
            # columns: eigst, dens, wkfxn (missing early in each SCF)
            dens.append(float(cols[1]) if len(cols) > 1 and cols[1] else np.nan)
            wf.append(float(cols[2]) if len(cols) > 2 and cols[2] else np.nan)
        m = re.match(r'Potential found to be ([\d.]+) V', line)
        if m:
            attempts.append((len(dens), float(m.group(1))))
    return np.array(dens), np.array(wf), attempts


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex='col',
                             facecolor=SURFACE)
    for j, (key, title, target) in enumerate(SYSTEMS):
        for tag, label, color in RUNS:
            dens, wf, att = parse(f'{tag}_{key}.txt')
            it = np.arange(1, len(dens) + 1)
            n_a, phi_a = np.array(att).T
            ax = axes[0, j]
            ax.semilogy(n_a, abs(phi_a - target), '-o', color=color, lw=2,
                        ms=6, mec=SURFACE, mew=1.5, label=label)
            ends = n_a.astype(int) - 1
            for row, y in [(1, dens), (2, wf)]:
                axes[row, j].plot(it, y, color=color, lw=1.2, alpha=0.9)
                axes[row, j].plot(it[ends], y[ends], 'o', color=color, ms=5,
                                  mec=SURFACE, mew=1)
        axes[0, j].set_title(title, color=INK, fontsize=10, loc='left')
        axes[0, j].axhline(0.01, color=MUTED, ls='--', lw=1)
        axes[0, j].text(1, 0.0105, 'tol 0.01 V', color=MUTED, fontsize=8,
                        va='bottom')
        axes[1, j].axhline(-4, color=MUTED, ls='--', lw=1)
        axes[2, j].axhline(-3, color=MUTED, ls='--', lw=1)
        axes[2, j].set_xlabel('cumulative SCF iterations')
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel('|phi - phi*|  (V)\nafter each outer attempt')
    axes[1, 0].set_ylabel('log10 density residual\n(dashed: final tol)')
    axes[2, 0].set_ylabel('log10 work-function change\n(last 3 SCF its; dashed: tol)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.955, 'Dots in rows 2-3 mark the end of each outer attempt '
             '(where N is updated). Inexact-Newton runs converge to looser '
             'tolerances on early attempts.', ha='center', color=MUTED,
             fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig('convergence.png', dpi=130, facecolor=SURFACE)
    print('wrote convergence.png')
