import bench_setup
atoms, calc = bench_setup.make('anchorcheck_Au40', {}, 4.0, 0.0,
                               keep_mixer='anchor')
atoms.get_potential_energy()
bm = calc.density.mixer.basemixers[0]
print('anchored', getattr(bm, '_n_anchored', 0), 'times; history length',
      len(bm.R_isG))
