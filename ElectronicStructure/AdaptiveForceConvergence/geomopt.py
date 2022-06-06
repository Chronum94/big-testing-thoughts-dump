import numpy as np

from gpaw import GPAW
from gpaw.scf import Criterion
from gpaw.forces import calculate_forces


from ase.build import mx2
from ase.optimize import BFGS
from ase.units import Ha, Bohr

atoms = mx2(vacuum=5) * [2, 2, 1]


class RelativeForces(Criterion):
    name = 'rel-forces'
    tablename = 'relforce'

    def __init__(self, tol, calc_last=True):
        self.tol = tol
        self.description = ('Maximum change in the atomic [forces] across '
                            'last 2 cycles: {:g} eV/Ang'.format(self.tol))
        self.calc_last = calc_last
        self.reset()

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if np.isinf(self.tol):  # criterion is off; backwards compatibility
            return True, ''
        with context.wfs.timer('Forces'):
            F_av = calculate_forces(context.wfs, context.dens, context.ham)
            F_av *= Ha / Bohr
        error = np.inf
        if self.old_F_av is not None:
            error = ((F_av - self.old_F_av)**2).sum(1).max()**0.5 / \
                np.max(np.linalg.norm(F_av, axis=1))
        self.old_F_av = F_av
        converged = (error < self.tol)
        entry = ''
        if np.isfinite(error):
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry

    def reset(self):
        self.old_F_av = None


adaptive_force_convergence = False
if not adaptive_force_convergence:
    convergence = {'forces': 1e-4}
    text_output = 'outbrute.txt'
else:
    convergence = {'eigenstates': 1e-5, 'density': 3e-4,
                   'custom': [RelativeForces(0.1)]},
    text_output = 'outadaptive.txt'

calc = GPAW(mode='pw',
            kpts=[6, 6, 1],
            convergence=convergence,
            txt=text_output
            )

atoms.rattle(0.1)
atoms.calc = calc

opt = BFGS(atoms)
opt.run(fmax=1e-3)
