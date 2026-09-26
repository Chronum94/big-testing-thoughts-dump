# Critical review: factorized radial experiments (2026-09-26)

Companion to `FACTORIZED_RADIAL.md`.

## Summary
Claim: a CP-factorised (canonical polyadic tensor decomposition) species-pair radial function generalises better than GMNN's dense pair table on sparse
multi-element data. Evidence: MAD dimers + trimers at 2.5k/5k/10k frames, 3–5 seeds for the main comparisons,
dense given a weight-decay scan and its best-validation checkpoint.

## Strengths
- Large effect (2–4×) relative to seed spread; holds in every |F_ref| bin (10k binned eval).
- Seen/unseen-pair split plus linear baseline.
- Train error reported → shows the gain is sharing, not regularisation.
- Dense baseline given a fair shot (wd scan, best-val checkpoint).

## Concerns

### Critical (could change conclusions)
1. **Init confounded with factorisation.** Best factorised model starts every pair near one shared radial
   (jitter 0.1); dense starts every pair independently random. Jitter 1.0 was ~2× worse → the win may be
   *shrinkage toward a shared function*, not low rank. Missing controls: (a) dense table initialised to a shared
   function + small jitter; (b) dense + penalty ‖W_ab − W̄‖² (hierarchical shrinkage). If (b) ≈ factorised, the
   claim is "shrinkage", not "low rank".
2. **Validation set does double duty.** Picks the checkpoint, tuned jitter, and is the reported metric →
   optimistic bias, favouring the factorised model (only one with a tuned hyperparameter). Need a held-out test split.
3. **Dense baseline is a strawman for sparse data.** Relevant comparisons are practical designs: neighbour-only
   v(Zn) (alchemical compression / MACE layer 1), symmetric u = v, concat-MLP (Allegro/Equiformer-style).

### Important
4. **Scope**: dimers/trimers only; says nothing about bulk/molecular environments.
5. **Single seeds + many comparisons**: rank-64, bs8, 10k are single-seed; metric splits chosen after seeing
   results → exploratory. Main claims need a pre-fixed comparison set on ≥3 seeds.
6. **Asymmetric tuning**: factorised got a jitter scan; dense got only wd. Dense `emb_lr`/`nn_lr` never tuned.
7. **"Unseen pair" is incidental**, from random sampling. A designed chemical hold-out (train Si–X, test Ge–X)
   tests the transfer the chemistry argument predicts.
8. **Mechanism asserted, not shown**: never checked whether u, v learn periodic-table structure.
9. **Body order vs data (added)**: see §Body order below.

### Minor
10. Absolute accuracy poor: low-force error ~0.7 eV/Å ≈ the forces themselves; MSE dominated by high-force frames.
11. Timing uncontrolled: different core counts and concurrency.
12. `per_element_force_rms_scale` computes sqrt(mean|F|), not sqrt(mean|F|²).

## Body order
GM contractions and the body order they can carry:

| contraction | moments | sums over | max body order |
|---|---|---|---|
| contr_0 | 1 | j | 2 |
| contr_1–3 | 2 | j, k | 3 |
| contr_4–7 | 3 | j, k, l | 4 |

- Dimers: 1 neighbour per centre → every contraction is a function of R(d) only (2-body info).
- Trimers: 2 neighbours per centre → everything is a function of (r_ij, r_ik, θ) (3-body info). The 3-moment
  contractions don't vanish (j = k = l self-terms are included), they just re-express 3-body information
  nonlinearly → redundant features.
- They have **no trainable parameters** themselves, so "data-starved" is about the readout: contr_4–7 are
  310 of 360 features at n_radial 5 (contr5/6/7 = 75/75/125, contr4 = 35), each with W1 weights fed
  degenerate/redundant inputs → extra readout capacity to overfit, no new information.
- Test: `n_contr: 4` keeps contr_0–3 (2- + 3-body, 50 features). Expect ≈ or better on this data. For bulk
  data the 4-body terms are needed, so this is a dataset-matched ablation, not a production change.
- Implication for the paper: on dimers/trimers the experiment only probes the pair channel and 3-body
  angular terms; 4-body behaviour is untested.

## Status (updated 2026-09-26)
- Concern 9 (body order): **done** — n_contr 4 ≈ 8 at 2.5k and 5k, 3 seeds (FACTORIZED_RADIAL §6.8).
- Concern 3 partial: **u-only / v-only done** — v-only ≈ CP; u-only much worse (§6.9). Symmetric u = v and
  concat-MLP baselines still open.
- Learning curve: 2.5k/5k (3 seeds), 10k/20k (1 seed) done (§6.11); rank 8 bias-limited at 20k. Rank 64 and
  12/7 basis at 20k running.
- Concern 8 partial: SV spectra of u, v, W done (§6.10); periodic-table structure check still open.
- Still open: 1 (shrinkage controls), 2 (test split), 6 (dense lr), 7 (designed hold-out), 4 (bulk), 10 (loss weighting).
- Capacity (§6.11): rank 64, 12/7 basis, wider readout [128,64,32] at 20k — none move forces; low-force floor
  ~0.72 common → loss weighting / force scaling / schedule now top suspects (new important concern).
- Rank 128 @2.5k better than rank 8, no overfitting (§6.12).
- Hybrid CP + Δ (§6.13): wd 0.1 memorises, 1.0 overfits seen, 10 ≳ rank 8; freeze-start no effect (reverted).
- **External validity (new, important)**: ethanol 3 seeds — rank 8 j0.1 ~36% worse than dense, underfits.
  Claim must be scoped to sparse multi-element regimes; hybrid/jitter choice is dataset-dependent.
- New minor: 20k has few unseen-pair val frames (pair quartiles 3/4/6) → unseen metrics noisier at scale; consider
  designed hold-out instead.

## Next steps (priority)

| P | Experiment | Settles |
|---|---|---|
| P1 | dense + shared init; dense + ‖W − W̄‖² penalty; dense lr scan. 3 seeds, 5k | 1, 6 |
| P1 | fixed train/val/test split; report test only | 2 |
| ~~P1~~ | ~~`n_contr: 4` for dense and factorised~~ (done) | 9 |
| P2 | ablations: ~~v only, u only~~ (done), u = v, concat-MLP | 3 |
| P2 | designed chemical hold-out (group-to-group transfer) | 7 |
| P3 | PCA/nearest neighbours of u, v vs group/period | 8 (cheap figure) |
| P3 | learning curve 2.5/5/10/20k — 1 seed done at 10k/20k; needs seeds | data- vs bias-limited |
| P4 | add bulk MAD subsets | 4 |
| P4 | φ(Z) prior + never-seen-element test | extension |
| P5 | force loss weighted 1/(\|F\|+c) or Huber | 10 |

## Overall
Moderate-to-strong evidence that factorised beats a naive dense table on sparse dimer/trimer data. Weak on
**why** (shrinkage vs low rank) and **how general**. P1 controls are cheap and could change the headline.
