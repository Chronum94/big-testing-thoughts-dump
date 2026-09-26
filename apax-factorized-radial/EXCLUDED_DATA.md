# Files not in this repo

Excluded because they're large (model checkpoints) or are training data. They live in
`~/scratch/factorized_radial/` on the original machine.

## Model directories (`models/`, ~481 MB)
One directory per run, named after the experiment (`models/<experiment>/`), containing checkpoints, `log.csv`
and `train_val_idxs.npz`. 89 runs: `ab_fr_*` (ethanol), `mad_*` (2.5k, first batch), `mad2k5_*`, `mad5k_*`,
`mad5kbs8_*`, `mad10k_*`, `mad20k_*`. Every number in `FACTORIZED_RADIAL.md` comes from these.
The `mad*_cmnn*` models need CMNN code (branch `cmnn-descriptor`, with the Bessel adapter) to reload.

## Training data
| File | Size | What |
|---|---|---|
| `mad-1.6-r2scan-train.xyz` | 703 MB | MAD-1.6 r2SCAN training set (source for all MAD subsets) |
| `mad_dimtri.traj` | 1.9 MB | 2,500 frames, MAD dimers+trimers (the "2.5k" set) |
| `mad_dimtri_5000.traj` | 3.3 MB | 5,000 frames |
| `mad_dimtri_10000.traj` | 5.9 MB | 10,000 frames |
| `mad_dimtri_20000.traj` | 12 MB | 20,000 frames |
| `etoh_aug3.traj` | 1.7 MB | ethanol, 1,651 frames incl. 100 stretched fragments |

The MAD subsets are reproducible: `python mad_dimtri_subset.py N` (filters `subset ∈ {dimers, trimers}` from
`mad-1.6-r2scan-train.xyz`, shuffles with seed 0, keeps the first N; each subset contains the smaller ones).
The 2.5k file was written as `mad_dimtri.traj` before the script took the `_{N}` suffix.

## Training logs
`*.out` (~5 MB): stdout/stderr of every training run (progress bars, warnings), named `<experiment>.out`.

## Also dropped
`_orphaned_ckpt_tmp/`: orbax temp/lock files left by a killed run (junk).

## Running
Scripts and configs use relative paths (`models/`, `*.traj`); run them from a directory that has `models/` and
the data files next to them. `run_*.sh` hard-code `~/scratch/factorized_radial`. Environment: apax branch
`factorized-radial` (local to the original machine, not pushed as of 2026-09-26), run via `run_cfg.py`.
