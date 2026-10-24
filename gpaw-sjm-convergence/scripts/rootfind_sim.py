"""Offline comparison of 1D root-finders for the SJM outer loop.

phi(N) curves come from our own data (Au/Pt constant-charge sweeps, Cu
attempt histories) plus a synthetic semiconductor. Each evaluation of
phi gets Gaussian noise: sigma = 1 mV for tight SCFs, or
max(1 mV, eta * |previous error|) for inexact ones (eta = 0.1), in which
case acceptance needs one extra tight evaluation at the same N.
All methods get the same first slope guess (0.65x the true capacitance at
the start, like our rigid-shift estimate).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

TOL = 0.01
SIG_TIGHT = 1e-3
ETA = 0.1
MAXEVAL = 20
NSEED = 300


# --- phi(N) curves ---------------------------------------------------------
def spline_curve(fname):
    n, _, phi = np.array(json.load(open(fname))).T
    cs = CubicSpline(n, phi)
    lo, hi = n.min(), n.max()
    s_lo, s_hi = cs(lo, 1), cs(hi, 1)

    def phi_of(x):  # linear extrapolation outside the sweep
        x = float(x)
        if x < lo:
            return float(cs(lo) + s_lo * (x - lo))
        if x > hi:
            return float(cs(hi) + s_hi * (x - hi))
        return float(cs(x))
    return phi_of


CU_PTS = np.array([  # tight (N, phi) points from base/est/anchor Cu logs
    (-0.18950, 4.18456), (-0.18813, 4.18077), (-0.16866, 4.02800),
    (-0.16699, 4.02459), (-0.16415, 3.99415), (-0.15947, 3.97301),
    (-0.15302, 3.91764), (-0.12678, 3.75322), (-0.12536, 3.74299),
    (-0.09806, 3.58139), (-0.07653, 3.47855), (-0.07558, 3.47865),
    (-0.03055, 3.26399), (0.0, 3.13249)])
_cu = np.polyfit(CU_PTS[:, 0], CU_PTS[:, 1], 3)


def semiconductor(phi0=5.0, sigma=0.05, n_s=1e-6, c_dl=0.15):
    """phi = phi0 - sigma*asinh(N/n_s) - N/c_dl, solved for given N
    (Boltzmann band tails around a ~1.4 eV gap, double layer in series)."""
    def phi_of(x):
        return phi0 - sigma * np.arcsinh(x / n_s) - x / c_dl
    return phi_of


CASES = [  # name, phi(N), N0, target
    ('Au 4.0 V', spline_curve('sweep_Au.json'), 0.0, 4.0),
    ('Pt 4.5 V', spline_curve('sweep_Pt.json'), 0.0, 4.5),
    ('Cu 4.0 V', lambda x: float(np.polyval(_cu, x)), 0.0, 4.0),
    ('Semi, conduction band (phi0-1.0)', semiconductor(), 0.0, 4.0),
    ('Semi, band edge (phi0-0.6)', semiconductor(), 0.0, 4.4),
]


# --- root-finders: each maps history -> next N -------------------------------
# history: lists N, phi, sig (measured), target, s0 (first slope guess)

def step_master(h, st):
    """master: linregress over last 4, slope mixed 0.5 with previous."""
    n, p = np.array(h['N'][-4:]), np.array(h['phi'][-4:])
    if len(n) > 1:
        s = np.polyfit(n, p, 1)[0]
        st['s'] = 0.5 * st['s'] + 0.5 * s
    return h['N'][-1] + (h['target'] - h['phi'][-1]) / st['s']


def step_branch(h, st):
    """sjm-convergence branch: weighted regression, first regressed slope
    replaces the guess, later ones mixed 0.5."""
    n, p = np.array(h['N'][-4:]), np.array(h['phi'][-4:])
    w = 1 / np.array(h['sig'][-4:])
    if len(n) > 1:
        s = np.polyfit(n, p, 1, w=w)[0]
        if s < 0:
            st['s'] = s if st['guess'] else 0.5 * st['s'] + 0.5 * s
            st['guess'] = False
    return h['N'][-1] + (h['target'] - h['phi'][-1]) / st['s']


def _anchored(h, m=4):
    n, p, sg = (np.array(h[k][-m:]) for k in ('N', 'phi', 'sig'))
    dn, dp = n[:-1] - n[-1], p[:-1] - p[-1]
    w = 1 / (sg[:-1] ** 2 + sg[-1] ** 2)
    return dn, dp, w


def step_anchored_lin(h, st):
    """Anchored weighted multisecant: line through the current point."""
    if len(h['N']) > 1:
        dn, dp, w = _anchored(h)
        s = (w * dn * dp).sum() / (w * dn * dn).sum()
        if s < 0:
            st['s'] = s
    return h['N'][-1] + (h['target'] - h['phi'][-1]) / st['s']


def step_anchored_iq(h, st):
    """Anchored weighted inverse quadratic: dN = a dphi + b dphi^2 through
    the current point; linear fallback if <3 points or curvature not
    resolved (spread of dphi < 5x the noise)."""
    df = h['target'] - h['phi'][-1]
    if len(h['N']) >= 3:
        dn, dp, w = _anchored(h)
        if np.ptp(np.append(dp, 0.)) > 5 * max(h['sig'][-3:]):
            A = np.c_[dp, dp ** 2] * np.sqrt(w)[:, None]
            a, b = np.linalg.lstsq(A, dn * np.sqrt(w), rcond=None)[0]
            if a < 0 and abs(b * df ** 2) < 0.5 * abs(a * df):
                st['s'] = 1 / a
                return h['N'][-1] + a * df + b * df ** 2
    return step_anchored_lin(h, st)


def step_iqi(h, st):
    """Classic inverse quadratic interpolation (last 3 points), secant /
    guess fallback."""
    if len(h['N']) >= 3:
        x, y = np.array(h['N'][-3:]), np.array(h['phi'][-3:])
        if len(set(np.round(y, 12))) == 3:
            t = h['target']
            return sum(x[i] * np.prod([(t - y[j]) / (y[i] - y[j])
                                       for j in range(3) if j != i])
                       for i in range(3))
    if len(h['N']) >= 2:
        s = (h['phi'][-1] - h['phi'][-2]) / (h['N'][-1] - h['N'][-2])
        if s < 0:
            st['s'] = s
    return h['N'][-1] + (h['target'] - h['phi'][-1]) / st['s']


def make_sidi(k):
    def step(h, st):
        """Sidi: Newton step with the derivative, at the current point, of
        the polynomial interpolating the last k+1 points."""
        m = min(k + 1, len(h['N']))
        if m >= 2:
            x, y = np.array(h['N'][-m:]), np.array(h['phi'][-m:])
            c = np.polyfit(x - x[-1], y, m - 1)
            s = c[-2]  # derivative at the current point
            if s < 0:
                st['s'] = s
        return h['N'][-1] + (h['target'] - h['phi'][-1]) / st['s']
    step.__doc__ = f'Sidi k={k}'
    return step


METHODS = [('master (regress + mix)', step_master),
           ('branch (weighted, replace guess)', step_branch),
           ('anchored linear', step_anchored_lin),
           ('anchored inverse-quadratic', step_anchored_iq),
           ('IQI (3-point)', step_iqi),
           ('Sidi k=2', make_sidi(2)),
           ('Sidi k=3', make_sidi(3))]


def true_slope(phi_of, x, d=1e-5):
    return (phi_of(x + d) - phi_of(x - d)) / (2 * d)


def run(method, phi_of, n0, target, inexact, rng):
    s0 = true_slope(phi_of, n0) / 0.65  # capacitance underestimated
    h = {'N': [], 'phi': [], 'sig': [], 'target': target}
    st = {'s': s0, 'guess': True}
    x, sig, nevals, errs = n0, (TOL if inexact else SIG_TIGHT), 0, []
    while nevals < MAXEVAL:
        val = phi_of(x)
        meas = val + rng.normal(0, sig)
        nevals += 1
        errs.append(abs(val - target))
        if abs(meas - target) < TOL:
            if sig > SIG_TIGHT:  # reconverge tightly at the same N
                meas = val + rng.normal(0, SIG_TIGHT)
                nevals += 1
                sig = SIG_TIGHT
                errs.append(abs(val - target))
            if abs(meas - target) < TOL:
                return nevals, errs, True
        h['N'].append(x)
        h['phi'].append(meas)
        h['sig'].append(sig)
        x_new = method(h, st)
        if not np.isfinite(x_new):
            return nevals, errs, False
        if inexact:
            sig = max(SIG_TIGHT, ETA * abs(meas - target))
        x = x_new
    return nevals, errs, False


def main():
    results = {}
    for inexact in [False, True]:
        mode = 'inexact' if inexact else 'tight'
        print(f'\n=== {mode} SCF noise; evaluations to |phi-phi*| < {TOL} V '
              f'(median / 90th pct / failures of {NSEED}) ===')
        print(f"{'method':34s}" + ''.join(f'{c[0][:18]:>20s}' for c in CASES))
        for mname, method in METHODS:
            row = []
            for cname, phi_of, n0, target in CASES:
                rng = np.random.default_rng(0)
                out = [run(method, phi_of, n0, target, inexact, rng)
                       for _ in range(NSEED)]
                ne = np.array([o[0] for o in out])
                ok = np.array([o[2] for o in out])
                results[(mode, mname, cname)] = out
                med = np.median(ne[ok]) if ok.any() else np.nan
                p90 = np.percentile(ne[ok], 90) if ok.any() else np.nan
                row.append(f'{med:5.1f}/{p90:4.1f}/{(~ok).sum():3d}')
            print(f'{mname:34s}' + ''.join(f'{r:>20s}' for r in row))
    return results


def plot(results):
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
              '#008300', '#6250d6']
    fig, axes = plt.subplots(2, len(CASES), figsize=(16, 6.5),
                             sharey=True, facecolor='#fcfcfb')
    for r, mode in enumerate(['tight', 'inexact']):
        for c, (cname, *_rest) in enumerate(CASES):
            ax = axes[r, c]
            for (mname, _), col in zip(METHODS, colors):
                out = results[(mode, mname, cname)]
                L = max(len(o[1]) for o in out)
                E = np.full((len(out), L), np.nan)
                for i, o in enumerate(out):
                    E[i, :len(o[1])] = o[1]
                    if o[2]:
                        E[i, len(o[1]):] = TOL / 10  # converged: floor
                med = np.nanmedian(np.log10(np.maximum(E, 1e-4)), axis=0)
                ax.plot(np.arange(1, L + 1), med, '-o', color=col, lw=1.8,
                        ms=3.5, label=mname)
            ax.axhline(np.log10(TOL), color='#6b6a64', ls='--', lw=1)
            ax.set_xlim(0.5, 12.5)
            ax.set_facecolor('#fcfcfb')
            ax.grid(color='#e6e5e0', lw=0.8)
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(cname, loc='left', fontsize=9)
            if r == 1:
                ax.set_xlabel('phi evaluation (SCF)')
        axes[r, 0].set_ylabel(f'{mode}: median log10 |phi - phi*|')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig('rootfind_sim.png', dpi=120, facecolor='#fcfcfb')


if __name__ == '__main__':
    plot(main())
