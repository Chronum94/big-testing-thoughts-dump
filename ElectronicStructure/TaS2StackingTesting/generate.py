from gpaw import GPAW, PW, MarzariVanderbilt, Mixer
from ase.io import read, write
from ase.build import stack
from scipy.stats.qmc import Sobol
import numpy as np


layer1 = read("TaS2_scell_geom_spair_05eTa-650eV-fullrelaxed_shifted_fineconvg.xyz")
layer2 = layer1.copy()
layer1.set_tags(np.ones(len(layer1)))
layer2.set_tags(np.ones(len(layer2)))

atomsiguess = []
sobolsample = Sobol(2, seed=2352349786).random(256)
for step in range(256):
    layer2.positions[:, :2] += sobolsample[step, 0] * layer2.cell[0, :2] + sobolsample[step, 1] * layer2.cell[1, :2]
    layer2.wrap()
    stacked = stack(layer1, layer2)
    atomsiguess.append(stacked)
write("layers.extxyz", atomsiguess)
    
    # stacked.write(f"honk{step}.xyz")
    
#     layer1.calc = calc
#     layer1.get_potential_energy()
