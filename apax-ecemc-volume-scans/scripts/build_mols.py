"""EC, EMC (RDKit ETKDG + MMFF), PF6- (ideal octahedron, P-F 1.61 A), Li+ -> mol/*.xyz"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
for name, smi in (("EC", "O=C1OCCO1"), ("EMC", "CCOC(=O)OC")):
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
    Chem.MolToXYZFile(m, f"mol/{name}.xyz")
d = 1.61; f = [(d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0), (0, 0, d), (0, 0, -d)]
open("mol/PF6.xyz", "w").write("7\nPF6-\nP 0 0 0\n" + "".join(f"F {x} {y} {z}\n" for x, y, z in f))
open("mol/Li.xyz", "w").write("1\nLi+\nLi 0 0 0\n")
# composition-breaking species (2026-10-21): DMC co-solvent; LiF and HF (LiPF6 decomposition / hydrolysis products)
m = Chem.AddHs(Chem.MolFromSmiles("COC(=O)OC")); AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m); Chem.MolToXYZFile(m, "mol/DMC.xyz")
open("mol/LiF.xyz", "w").write("2\nLiF\nLi 0 0 0\nF 1.564 0 0\n")
open("mol/HF.xyz", "w").write("2\nHF\nF 0 0 0\nH 0.917 0 0\n")
