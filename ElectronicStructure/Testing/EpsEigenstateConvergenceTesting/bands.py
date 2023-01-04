from gpaw import GPAW


bands_to_converge = 196

for i, convergence in enumerate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    calc = GPAW("outgs.gpw").fixed_density(
        kpts={"size": [4, 4, 12]},
        convergence={"bands": bands_to_converge, "eigenstates": convergence},
        nbands=int(1.2 * bands_to_converge),
        txt=f"outbands{i}.txt",
    )
    calc.write(f"outbands{i}.gpw", "all")
