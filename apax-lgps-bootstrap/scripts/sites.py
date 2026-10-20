"""Expand COD 7215191 (Li10GeP2S12, P4_2/nmc origin 1, 298 K) into its partial-occupancy site list."""
import numpy as np
from ase.spacegroup import crystal

CELL = (8.719, 8.719, 12.639, 90, 90, 90)
# label, species, fractional xyz, occupancy (from the CIF)
SITES = [("Li1", "Li", (0.267, 0.273, 0.194), 0.466), ("Li2", "Li", (0.0, 0.5, 0.9454), 0.86),
         ("Li3", "Li", (0.248, 0.248, 0.0), 0.74), ("Li4", "Li", (0.0, 0.0, 0.251), 0.81),
         ("GeP", "Ge", (0.0, 0.5, 0.69154), 0.5), ("P2", "P", (0.0, 0.0, 0.5), 1.0),
         ("S1", "S", (0.0, 0.18857, 0.40707), 1.0), ("S2", "S", (0.0, 0.29557, 0.09727), 1.0),
         ("S3", "S", (0.0, 0.69901, 0.79168), 1.0)]


def expand():
    """Return (Atoms with every site filled, label array, occupancy array)."""
    a = crystal([s[1] for s in SITES], [s[2] for s in SITES], spacegroup=137, setting=1, cellpar=CELL)
    lab = np.array([SITES[t][0] for t in a.get_array("spacegroup_kinds")])
    occ = np.array([SITES[t][3] for t in a.get_array("spacegroup_kinds")])
    return a, lab, occ


if __name__ == "__main__":
    a, lab, occ = expand()
    for l in dict.fromkeys(lab):
        m = lab == l
        print(f"{l:4s} n={m.sum():2d} occ={occ[m][0]:.3f} -> {m.sum() * occ[m][0]:.2f} atoms")
    li = lab.astype(str).tolist()
    print("total Li expected", sum(occ[i] for i in range(len(a)) if lab[i].startswith("Li")))
    d = a.get_all_distances(mic=True); np.fill_diagonal(d, 9)
    L = np.flatnonzero(np.char.startswith(lab.astype(str), "Li"))
    dl = d[np.ix_(L, L)]
    print("closest Li-Li site pairs:", sorted(set(np.round(dl[dl < 2.3], 2))))
