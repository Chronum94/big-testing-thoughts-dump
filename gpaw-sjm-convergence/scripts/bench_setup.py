"""Shared SJM test system: Au(111) 1x1x3 in implicit water."""
import types

import numpy as np
from ase.build import fcc111
from ase.units import Ha
from gpaw import FermiDirac
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.convergence_criteria import WorkFunction


def _quick_reinitialization_keep_mixer(self):
    """OldSJM._quick_reinitialization without self.density.mixer.reset()."""
    if self.density.nct_G is None:
        self.initialize_positions()
    self.density.reset()
    self._set_atoms(self.atoms)
    self.wfs.initialize(self.density, self.hamiltonian, self.spos_ac)
    self.wfs.eigensolver.reset()
    if self.scf:
        self.scf.reset()


# Lists of an MSR1 base mixer that hold one entry per evaluated input.
_PAIR_LISTS = ['nt_isG', 'R_isG', 'mR_isG', 'D_iasp', 'dD_iasp', 'mD_iasp']


def _translate(x_i, last, anchor):
    """Shift every history entry by (anchor - last); entries are arrays or
    lists of per-atom arrays."""
    if isinstance(anchor, list):
        return [[x - xl + a for x, xl, a in zip(x_a, last, anchor)]
                for x_a in x_i]
    return [x - last + anchor for x in x_i]


def _anchored_mix_density(self, nt_sG, D_asp, g_ss=None):
    """MSR1 mix_density that, once the first (input, residual) pair at the
    new electron count exists, prepends the old pairs translated so that
    the last old pair coincides with it (secant differences preserved)."""
    old = self._pending_history
    if old is not None and len(self.R_isG) == 1 and len(self.nt_isG) == 2:
        for name in _PAIR_LISTS:
            x_i = old[name]
            new = _translate(x_i[:-1], x_i[-1], getattr(self, name)[0])
            getattr(self, name)[0:0] = new
        self._pending_history = None
        self._n_anchored = getattr(self, '_n_anchored', 0) + 1
    return type(self).mix_density(self, nt_sG, D_asp, g_ss)


def _quick_reinitialization_anchor_mixer(self):
    """OldSJM._quick_reinitialization, but the mixer history is re-anchored
    at the new electron count instead of discarded."""
    for bm in self.density.mixer.basemixers:
        n = len(bm.R_isG)
        bm._pending_history = ({name: list(getattr(bm, name)[:n])
                                for name in _PAIR_LISTS} if n > 1 else None)
        bm.mix_density = types.MethodType(_anchored_mix_density, bm)
    # The original resets the mixer (lists only) before the first new mix.
    type(self)._quick_reinitialization(self)


class AdaptiveWorkFunction(WorkFunction):
    """Work-function criterion for non-final SJM attempts: converged when
    the spread of the last n_old work functions is below
    max(tol, eta * |phi - target|), phi being the top-side value."""

    def __init__(self, tol, n_old, target, eta):
        super().__init__(tol, n_old)
        self.target = target
        self.eta = eta
        self.description = (f'Adaptive work function: spread < max({tol:g}, '
                            f'{eta:g} |phi - phi*|) eV')

    def __call__(self, context):
        wf = Ha * np.array(context.ham.get_workfunctions(context.wfs))
        self._old.append(wf)
        if len(self._old) < self._old.maxlen:
            return False, ''
        error = max(np.ptp(self._old, axis=0))
        threshold = max(self.tol, self.eta * abs(wf[1] - self.target))
        return error < threshold, f'{np.log10(error):+5.2f}'


def _set_scf_factor_adaptive(self, factor):
    """OldSJM._set_scf_factor, but loose attempts use coarse density /
    eigenstates / energy tolerances and the adaptive work-function
    criterion instead of scaled tolerances."""
    type(self)._set_scf_factor(self, factor)
    if self.scf is None:
        return
    criteria = self.scf.criteria
    p = self.parameters['sj']
    if factor > 1.:
        criteria['density'].tol = self._adaptive['density']
        criteria['eigenstates'].tol = self._adaptive['eigenstates']
        criteria['energy'].tol = 1e3
        if not isinstance(criteria['work function'], AdaptiveWorkFunction):
            self._tight_wf = criteria['work function']
            criteria['work function'] = AdaptiveWorkFunction(
                self._tight_wf.tol, self._tight_wf.n_old,
                p.target_potential, self._adaptive['eta'])
            criteria['work function'].reset()
    elif isinstance(criteria['work function'], AdaptiveWorkFunction):
        criteria['work function'] = self._tight_wf
        self._tight_wf.reset()


def _inexact_factor_always_loose(self, dphi):
    """Every non-final attempt is loose (factor >= 2, also attempt 0); the
    factor still sets the regression weights."""
    factor = type(self)._inexact_factor(self, dphi)
    if self.parameters['sj'].inexact_newton is None:
        return factor
    return max(2., factor)


def make(tag, extra, target=4.0, n0=0.0, element='Au',
         unsolv_backside=False, keep_mixer=False, adaptive=None):
    if 'base' in tag.split('_'):
        from sjm_master import SJM, SJMPower12Potential
    else:
        from gpaw.solvation.sjm import SJM, SJMPower12Potential
    atoms = fcc111(element, size=(1, 1, 3))
    atoms.cell[2][2] = 15
    atoms.translate([0, 0, 6 - min(atoms.positions[:, 2])])

    sj = {'target_potential': target, 'excess_electrons': n0,
          'jelliumregion': {'top': 14.5}, 'tol': 0.01}
    sj.update(extra)

    calc = SJM(mode='fd', kpts=(2, 2, 1), xc='PBE',
               occupations=FermiDirac(0.1), txt=f'{tag}.txt', sj=sj,
               cavity=EffectivePotentialCavity(
                   effective_potential=SJMPower12Potential(
                       u0=0.180, unsolv_backside=unsolv_backside),
                   temperature=298.15, surface_calculator=GradientSurface()),
               dielectric=LinearDielectric(epsinf=78.36),
               interactions=[SurfaceInteraction(surface_tension=0.00114843767916)])
    if adaptive:
        calc._adaptive = adaptive
        calc._set_scf_factor = types.MethodType(_set_scf_factor_adaptive,
                                                calc)
        calc._inexact_factor = types.MethodType(
            _inexact_factor_always_loose, calc)
    if keep_mixer == 'anchor':
        calc._quick_reinitialization = types.MethodType(
            _quick_reinitialization_anchor_mixer, calc)
    elif keep_mixer:
        calc._quick_reinitialization = types.MethodType(
            _quick_reinitialization_keep_mixer, calc)
    atoms.calc = calc
    return atoms, calc
