from gpaw.response.df import DielectricFunction


for i in range(5):
    df = DielectricFunction(
        calc=f"outbands{i}.gpw",
        domega0=0.1,
        ecut=100,
        eta=0.2,
        rate=0.5,
        nbands=196,
        nblocks=12,
        txt=f"response{i}_100eV.txt",
    )
    df.get_dielectric_function(
        direction="x",
        filename=f"eps{i}_100eV.csv",
    )
