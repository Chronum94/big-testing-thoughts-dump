> **Note:** this study was carried out with Claude (Anthropic's AI assistant) under close human oversight.
> None of the conclusions here are final.

# GMNN descriptor cost vs accuracy on the GPU (apax): radial channels, tensor channels, l_max, packed moments

Which GM descriptor knobs buy accuracy, and what each costs in MD wall time on a GPU.

- **A** = `n_radial`: scalar (l = 0) radial channels.
- **B** = `n_radial_tensor`: radial channels feeding the l = 1–3 tensor moments.
- **l ≤ 2** ("L3" below): `drop_contractions: [3, 6, 7]`. This removes the rank-3 moment and its contractions, and XLA drops the m3 scatter with it.
- **Packed moments:** `packed_moments: true` stores only the unique components of the symmetric moments: 3 + 6 + 10 = 19 floats per B channel instead of 3 + 9 + 27 = 39. It is an exact reformulation with no parameter change.

## Requirements: custom apax branch, GPU setup

- **Code:** branch `shallow-ensemble-uq` of [github.com/Chronum94/apax](https://github.com/Chronum94/apax/tree/shallow-ensemble-uq). The training runs used commit `144a9065`.
  - Packed moments (`b318a6da`) and the padded-atom metric fix (`46644964`) are in the pushed branch.
  - `144a9065` itself (the pair-envelope init fix) is not pushed yet.
- **Hardware:** AMD RX 9070 XT, ROCm in a podman container on WSL2, JAX 0.10.0 in the container.
  - The branch carries the AMD atomics work: centre-sorted neighbour lists, a patched scatter-add transpose, and CUDA/HIP graphs off.
  - `XLA_FLAGS=--xla_gpu_enable_command_buffer=` was set for all runs.
  - `costmap/run_costmap.sh` and `packed_equiv/run_cpu.sh` show the container invocations. The paths are local to the machine the study ran on.

## Data (not included)

- **Accuracy runs:** molecular crystals, the `shiftml_molcrys` subset of MAD-1.6.
  - Random splits: 1000 train, 200 val, 500 test structures, 98–103 atoms on average (10–200).
  - `splits.json` lists the `frame_id` of every structure in each split. The splits are disjoint.
- **Cost map:** one 480-atom cell of EMIM-BF4 from the apax paper data ([apax-hub/apax_paper](https://github.com/apax-hub/apax_paper), `r1000_training` branch, final training set), timed as-is and as a 3×3×3 box of 12,960 atoms. Its model settings are in `costmap/emim_r55.yaml`.

## Setup

- **Model** (`configs/`):
  - GMNN with 12 Bessel functions, r_max 5 Å, CP radial (rank 8), readout [64], per-element isolated-atom shift, pair envelope (trainable).
  - A = 12 and B = 4 unless stated; this is the "L0" default.
- **Training:** Muon (lr 0.02, wd 0.03), batch 4, 400 epochs on one cyclic cosine period, energy and force MSE with weight 1 each.
- **Repeats:** one seed per config. The B4 baseline was run with 3 seeds, and seeds 1 and 2 agree to 0.1 meV/Å on val F.
- **Metric:** val MAE at the last epoch. Force MAE is per component, over real atoms only. Energy MAE is eV/structure.
- **Cost:** MD-like jitted energy + forces on the 12,960-atom box, about 66 pairs/atom at 5.5 Å. Times are medians of repeated calls; run-to-run noise is about ±10 %.

## Salient results

**1. Accuracy (molcrys val, meV/Å; MD ms unpacked)**

| config | MD ms | val F | dF vs L0 | dE vs L0 |
|---|---|---|---|---|
| L3 A12/B4 | 10.0 | 148.6 | +11.8 % | +13.4 % |
| L3 A24/B4 | 11.4 | 145.0 | +9.0 % | +16.2 % |
| L3 A32/B4 | 11.3 | 145.3 | +9.3 % | +16.3 % |
| L3 A12/B6 | 10.8 | 139.9 | +5.2 % | +8.6 % |
| L3 A12/B10 | 20.1 | 135.3 | +1.8 % | −0.5 % |
| B2 | 9.8 | 150.4 | +13.1 % | +20.1 % |
| B3 | 13.3 | 138.8 | +4.4 % | +5.1 % |
| **L0 A12/B4** | 15.65 | 132.9 | – | – |
| **B5** | 16.3 | 128.2 | −3.5 % | −12.0 % |
| A16/B5 | 16.3 | 128.5 | −3.3 % | −12.4 % |
| B6 | 24.55 | 127.2 | −4.4 % | −8.3 % |
| B8 | 35.7 | 121.7 | −8.4 % | −22.5 % |

- **A is saturated:** A12 → 24 → 32 at l ≤ 2, and A12 → 16 at B5, change val F by at most 3.6 meV/Å, and the energy gets worse.
- **B carries the accuracy** and has not flattened by B8.
- **l ≤ 2 has a ceiling:** even at B10 (+28 % MD cost) it does not reach L0.
- **Contraction drops at A12/B4**, from `logs/v2nc_s*` and `logs/v2L2_s*`:
  - dropping c4/c6: +2.7 % F;
  - dropping c4/c6/c7: +8.4 % F.

**2. Linear model of val F in A and B** (`fit_ab.py`, meV/Å)

| family | model | const | A | B | rms / max resid |
|---|---|---|---|---|---|
| l ≤ 3 | linear | 163.6 | −0.82 /channel | −4.41 /channel | 3.07 / 5.43 |
| l ≤ 3 | log2 | 174.3 | −3.3 /doubling | **−14.0 /doubling** | **1.39** / 2.03 |
| l ≤ 2 | linear | 155.7 | −0.08 /channel | −2.03 /channel | 1.58 / 2.66 |
| l ≤ 2 | log2 | 173.5 | −1.8 /doubling | **−9.8 /doubling** | **1.10** / 1.82 |

- **Log2 fits best.** A plain linear fit under-predicts B2 and over-predicts B8.
- **The l ≤ 3 A coefficient is not identified:** only one run varies A (A16 vs A12 at B5: +0.3 meV/Å). Read it as about 0.
- **Practical form:** val F ≈ 133 − 14·log2(B/4) meV/Å, for l ≤ 3 and A ≥ 12, within B 2–8.

**3. Cost map** (`costmap/costmap.jsonl`; MD ms, 12,960 atoms, unpacked)

| A \ B | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 8 | 9.9 | 12.2 | 15.4 | 20.6 | 29.6 |
| 12 | 9.8 | 13.3 | 15.4 | 24.4 | 35.7 |
| 16 | 10.1 | 12.3 | 18.1 | 21.0 | 32.2 |
| 24 | 10.9 | 13.6 | 16.1 | 22.1 | 31.5 |
| 32 | 11.1 | 14.1 | 17.8 | 24.8 | 30.8 |

- **B sets the cost.** B2 → B8 at A12 is 3.6× slower; A8 → A32 at B4 is 16 % slower.
- **One B channel costs about 40 A channels**, close to the 39:1 ratio in floats scattered per pair. MD here is limited by memory traffic, not FLOPs. The per-atom contractions grow as B³ but are nearly free in wall time.
- **l ≤ 2 at A12/B4:** 9.5 ms (−38 %).
- **Training step** on the 480-atom cell: 2.3–4.5 ms across the grid, dominated by overhead at that size.

**4. Packed moments** (exact; MD ms, same session)

| B (A12, l ≤ 3) | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| unpacked | 9.8 | 13.3 | 15.6 | 15.8 | 23.8 | 33.5 | 45.4 |
| packed | 10.3 | 11.0 | 12.7 | 13.6 | 16.9 | 22.9 | 29.4 |
| speedup | 0.95× | 1.20× | 1.23× | 1.16× | 1.41× | 1.47× | 1.54× |

- **Exact on trained checkpoints, no retraining** (`packed_equiv/equiv.py`; B5, B6 and L3 B10 on 20 val structures each): max |dF| ≈ 3e-6 eV/Å, max |dE| ≈ 1e-7 eV/atom, identical MAEs.
- **The gain grows with B.** Packing shrinks only the B part of the per-pair data. At A12 it removes 20B / (12 + 39B) of the scattered floats: 22 % at B2, 47 % at B10, approaching 51 %.
  - **Small B:** at B2 the packed contractions' overhead dominates.
  - **B5 dip:** unpacked B5 happens to sit just before the unpacked B5 → B6 jump (+51 %). Packing shrinks that jump to +24 %.
- **l ≤ 2 gains about 1.0×,** since there is no m3 to pack.
- **Training step:** about 5 % faster.
- **Consequences:**
  - Packed B5 (13.6 ms, 128.2) beats unpacked L0 B4 (15.65 ms, 132.9) on both axes.
  - Packed B3 (11.0 ms, 138.8) matches l ≤ 2 B6 (10.8 ms, 139.9) at equal cost, with better energy. So l ≤ 2 is off the front once packed.
- **Plots:** `figs/pareto.png` (unpacked) and `figs/pareto_packed.png`.

**5. Choosing the next point with a GP** (`figs/next_point.py`)

A GP on log val F over (log2 A, log2 B, l), with expected improvement against the Pareto front at measured MD cost. It interpolated along B well (B3 predicted 138.9 ± 1.9, measured 138.8), but missed L3 B10 by more than 2σ. With about 10 points, mostly in 1-D slices, it is not a reliable picker.

## Caveats

- **One seed per config,** except the B4 baseline.
- **Not converged:** 400 epochs of one cosine period, and the curves were still falling at the end.
- **One dataset:** molecular crystals. A saturating may not carry over to bulk, where A24 helped in earlier CPU runs.
- **Possible JAX 0.10 artefact:** part of the unpacked B5 → B6 jump may be the JAX ≥ 0.10 slowdown on unflagged transposed gathers reported in [lab-cosmo/jax-pme#39](https://github.com/lab-cosmo/jax-pme/pull/39). The neighbour-side gather in apax `distances.py` is still unflagged. A second JAX version would separate layout from version.

## Files

- `configs/`: all training configs; `make.py` generated the v2 B4 / no-c4/c6 arms. The `p1_*` configs are copies of `v2B4_s1` with A, B or drops changed.
- `logs/`: `log.csv` of every run (`<run>.csv`).
- `costmap/`: `costmap.py` (grid timing), `run_costmap.sh` (GPU container), `costmap.jsonl` (all rows; packed and extra rows carry a `batch` timestamp), `emim_r55.yaml`.
- `packed_equiv/`: `equiv.py` and `run_cpu.sh` (CPU container), plus `cpuload.py` (loads GPU-saved checkpoints on CPU).
- `figs/`: `pareto.py` (`--packed` for the packed timings), `next_point.py`, and the PNGs.
- `fit_ab.py`: the A/B fits in table 2.
- `splits.json`: frame IDs of the dataset splits.
