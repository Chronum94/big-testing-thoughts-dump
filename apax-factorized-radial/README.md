> **Note:** this study was carried out with Claude (Anthropic's AI assistant) under close human oversight.
> None of the conclusions here are final.

# Factorized species-pair radial functions for GMNN (apax)

Replacing GMNN's dense (119 × 119 × n_radial × n_basis) species-pair radial coefficient table with a low-rank
CP (canonical polyadic, a.k.a. CANDECOMP/PARAFAC tensor decomposition) factorisation over per-element embeddings, tested on MAD-1.6 dimers/trimers and ethanol.

## Requirements: custom apax branch

The configs here use options that don't exist in upstream apax (`radial_rank`, `radial_emb_jitter`,
`radial_factor_mode`, `radial_residual`, optimizer `residual_wd`). You need the **`factorized-radial` branch of
[github.com/Chronum94/apax](https://github.com/Chronum94/apax)**. As of 2026-09-26 that branch has not been
pushed yet; it lives only on the original machine.

## Method in one line

```
radial_r(r; Z_centre, Z_nbr) = (1/rank) Σ_k u_k(Z_centre) · v_k(Z_nbr) · (C_k · basis(r))_r · cutoff(r)
```
u, v: (n_species, rank) per-element embeddings initialised at 1 + jitter·N(0,1) (pairs start near one shared
radial function); C: (n_basis, rank·n_radial) shared core. Rank 8: ~2k species parameters vs 566k for the dense table.

## Salient results

Data: MAD-1.6 dimers + trimers, random subsets of 2.5k / 5k / 10k / 20k frames (102 elements, pairs mostly seen
1–2 times at ≤10k). GMNN, 8 Bessel / 5 radial, readout [64, 32, 16], 200 epochs, per-element regression shift,
no repulsion term. Validation is split into frames whose element pairs all appear in training ("seen") and frames
with at least one unseen pair. MAE: E in eV/structure, F in eV/Å per component, best-validation checkpoint.

**1. Factorised vs dense, 5k, mean ± std over seeds (dense 3 seeds, factorised 3 seeds)**

| Model | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|
| per-element linear fit + zero force | 3.13 | 3.45 | 3.18 | 4.09 |
| dense pair table | 2.43±0.08 | 2.20±0.15 | 2.98±0.07 | 2.98±0.15 |
| rank 8, jitter 1.0 | 2.00±0.16 | 1.25±0.11 | 1.99±0.13 | 1.34±0.13 |
| **rank 8, jitter 0.1** | **1.58±0.01** | **0.59±0.01** | **1.57±0.05** | **0.73±0.08** |

About 4× lower unseen-pair force error. Dense memorises (train F → 0.02) and generalises barely better than the
linear baseline on energy. Weight decay on dense (1e-5 to 1e-1) doesn't help. Factorised fits the training set as
well as dense does, so the gain is **sharing across pairs, not regularisation**. Starting all pairs near a shared
radial function (small jitter) is a strong prior on sparse data.

**2. Which factor matters (5k, rank 8, jitter 0.1, 3 seeds)**

| Model | seen F | unseen E | unseen F |
|---|---|---|---|
| CP u(Z_centre)·v(Z_nbr) | 0.59±0.01 | 1.57±0.05 | 0.73±0.08 |
| neighbour only v(Z_nbr) | 0.60±0.04 | 1.58±0.10 | 0.78±0.07 |
| centre only u(Z_centre) | 1.12±0.14 | 2.24±0.09 | 1.35±0.11 |

The neighbour embedding carries the gain (alchemical-compression / MACE-layer-1 structure). Centre-only can't see
which elements its neighbours are.

**3. Learning curve (rank 8, jitter 0.1; 2.5k = 3-seed mean, others seed 1)**

| Frames | train F | seen F | unseen E | unseen F |
|---|---|---|---|---|
| 2.5k | 0.60 | 0.68 | 1.86 | 1.04 |
| 5k | 0.34 | 0.57 | 1.52 | 0.69 |
| 10k | 0.31 | 0.47 | 1.26 | 0.51 |
| 20k | 0.29 | 0.35 | 1.06 | 0.45 |

Dense stays at unseen F ≈ 2.9–3.0 from 2.5k to 10k. By 20k rank 8 is mostly bias-limited (train–val gap 0.06).

**4. What doesn't move forces at 20k (seed 1)**

| Model | train F | seen F | unseen E | unseen F | \|ΔF\| at \|F_ref\| < 1 |
|---|---|---|---|---|---|
| rank 8 | 0.29 | 0.35 | 1.06 | 0.45 | 0.72 |
| rank 64 | 0.22 | 0.37 | **0.81** | 0.45 | 0.72 |
| rank 8, 12 Bessel / 7 radial | 0.27 | 0.37 | 1.03 | 0.40 | 0.75 |
| rank 8, readout [128, 64, 32] | 0.29 | 0.35 | 1.08 | 0.39 | 0.74 |
| CMNN (message-passed embeddings) | 0.40 | 0.46 | 1.08 | 0.57 | 0.77 |

More rank buys energy, not forces. The low-force floor (~0.72) is identical across capacity increases, which points
to loss weighting / force scaling / schedule. Static CP beats the message-passing embedding here at lower cost.

**5. Adding capacity at 2.5k (seed 1)**

| Model | train F | seen F | unseen F |
|---|---|---|---|
| rank 8 | 0.65 | 0.58 | 0.90 |
| rank 128 | 0.36 | **0.45** | **0.81** |
| rank 8 + dense per-pair residual Δ, wd 0.1 | 0.12 | 0.99 | 1.60 |
| rank 8 + Δ, wd 10 | 0.49 | 0.56 | 0.87 |

Rank 128 doesn't overfit even with 1–2 samples per pair (implicit low-rank bias of the factorised
parametrisation). A per-pair residual needs strong decay under Adam, which otherwise gives rarely seen pairs
full-size steps.

**6. Not a universal win: ethanol (3 elements, data-rich), 3 seeds, 40 epochs**

| Model | val E (meV) | val F (meV/Å) |
|---|---|---|
| dense | 25.2±0.9 | 44.4±1.8 |
| rank 8, jitter 0.1 | 39.6±8.2 | 60.5±9.9 |

With few elements and plenty of data per pair, the shared-radial prior underfits. The claim is scoped to sparse,
multi-element data.

**Also found:** removing 4-body contractions (`n_contr: 4`) changes nothing on dimers/trimers; apax's default
regression-shift ridge (`energy_regularisation: 1.0`) gives keV errors on all-electron energies (use 1e-6).

## Files

- `FACTORIZED_RADIAL.md` — full write-up: motivation, literature, method, cost/precision analysis, all experiments.
- `CRITIQUE.md` — critical review, open concerns, prioritised next steps.
- `EXCLUDED_DATA.md` — what isn't in this repo (model checkpoints, training data, logs) and how to regenerate it.
- Scripts: `mad_dimtri_subset.py` (data subsets), `eval_multi.py`, `eval_fbins.py`, `eval_mad5k.py`,
  `eval_wd_s1.py`, `core_rank.py` (evaluation/analysis), `run_*.sh` (launchers), `config_*.yaml` (every run).
