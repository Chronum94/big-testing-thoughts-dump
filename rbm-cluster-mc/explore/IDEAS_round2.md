# Round 2: next experiments (proposals, not findings)

arXiv IDs checked via the arXiv API on 2026-10-14. The Potts L=10 follow-up to §8 is already running.

**Core construction (#1–#4).** Add bond gates n_b∈{0,1}:
p(s,h,n) ∝ e^{−E_RBM(s,h)} ∏_b[1−n_b + n_b(e^{K_b}−1)δ_b(s)], with δ_b = "bond satisfied".
- Summing out n gives log p_θ = −F + ΣK_bδ_b, so the exact acceptance uses **d = −βE + F − ΣK_bδ_b**.
- Given s, h and n are independent; P(n_b=1) = δ_b(1−e^{−K_b}).
- Given (h,n), each cluster C draws a shift k ∝ exp Σ_{i∈C} ℓ_i((s_i+k) mod q), where ℓ = b+W·h.
- This is an FK step with RBM-biased cluster colours: genuine cluster moves, still reversible.
- **Code:** `gated_mh_step` in rbm.py, ~30 lines.
- **Prior art:** generalised bond probabilities come from Niedermayer (PRL 61, 2026 (1988); no arXiv) and Kawashima & Gubernatis (cond-mat/9411110). Wang (1702.08586) got SW from a three-body Boltzmann machine. New here: RBM-biased colouring and the learned residual.

## 1. A continuous bridge from local moves to SW
- **Hypothesis:** for Ising, set K_b = 2λβ|J_b|. The residual target π/p_gate is then Ising with couplings (1−λ)J, so `exact_ising_rbm(lat, (1−λ)β)` gives acceptance 1 for every λ.
  - λ=0 is §7 (z≈2.4); λ=1 is SW.
  - Prediction: z stays near 2 until the gate clusters percolate, then drops sharply.
- **Experiment:** 2D Ising at T_c, L=8–16, λ∈{0, .25, .5, .75, .9, 1}, 3 seeds. Measure τ_|m|, cluster size, τ×ms.
- **Falsified if:** λ=1 gives z>0.5 (a bug), or τ(λ) is flat.
- **Cost:** ~10 min.
- **Refs:** 1702.08586; cond-mat/9411110.

## 2. Known-answer disorder benchmark: bond dilution
- **Located evidence:** Kole et al. (2107.08534) report Wolff z≈1.75 on the 2D bond-diluted Ising model (isolated spins rarely visited); SW has z≈0.09 at p=0.6.
- **Hypothesis:** gated λ=1 reproduces this. "Wolff + sparse-RBM blocks" helps no more than a "Wolff + Metropolis sweep" null.
- **Code:** none in rbm.py.
- **Experiment:** p=0.6 at T_c(p), L=8–16.
- **Falsified if:** RBM blocks beat the null by more than 1.5× in τ×cost.
- **Cost:** ~10 min.
- **Refs:** 2107.08534.

## 3. Frustration-aware gates, 2D ±J
- **Hypothesis:** FK clusters percolate in the glass phase. Per-bond λ_b, with residual `exact_ising_rbm` on (1−λ_b)J, keeps acceptance = 1.
  - Set λ_b = λ_u if both of the bond's plaquettes are unfrustrated, else λ_f.
  - Prediction: an interior optimum beats Metropolis on τ_E×cost.
- **Code:** a scaled-J lattice copy (~3 lines) plus a scan.
- **Experiment:** L=8–16 at T=1.0 and 0.6, 4 disorder seeds. Compare against Houdayer (~30-line outside script).
- **Falsified if:** the optimum is at λ≈0 or 1, or the best point loses to Metropolis.
- **Cost:** ~20 min.
- **Refs:** cond-mat/0101116; 1501.05630.

## 4. Potts ±J (q=3): gates where FK works, the RBM where it fails
- **Hypothesis:** Potts antiferromagnetic bonds have no two-state FK. So gate only the J>0 bonds (K_b = λβJ_b) and let a sparse RBM learn the residual.
  - Train by PCD with `gated_mh_step` negative chains. The gradient for (W,b,c) is unchanged.
  - Prediction: the decay of acceptance with N weakens as λ grows, because the residual is a weaker, "hotter" model.
- **Code:** a `neg_step=` argument in `fit` (~4 lines).
- **Experiment:** L=6–12, λ∈{0, .5, 1}, plus §8 nulls.
- **Falsified if:** log(acceptance) vs N has the same slope at every λ.
- **Cost:** ~20 min.
- **Refs:** 1702.08586; 1610.03137.
- **Note:** a bounded search found no learned cluster move for Potts glasses (not proof of novelty).

## 5. Log-variance on data ∪ proposals (repairs §4)
- **Hypothesis:** apply the Var[d] loss to a pooled batch {x, Gibbs(x)}, with the proposals under stop-gradient.
  - On model samples this is the reverse-KL gradient: it penalises p_RBM ≫ π.
  - Prediction: dshift→0 and block acceptance exceeds PCD.
- **Code:** ~3 lines in `var_loss.py`.
- **Experiment:** the NOTES §4 table plus pooled arms.
- **Falsified if:** sd_prop falls but acceptance does not beat PCD.
- **Cost:** ~5 min.
- **Refs:** 2010.10436; 1610.03137.

## 6. Maximise expected acceptance × jump (REINFORCE)
- **Hypothesis:** log p(h|s) + log p(s'_A|h) is tractable, so the score-function gradient of G = E[min(1, e^{Δd})·|s'≠s|] is unbiased.
  - Prediction: on r=2 blocks, G is more than 1.5× PCD's, with larger |C|.
- **Code:** a ~25-line `fit_accept` function.
- **Experiment:** Ising L=8 at T=2.5 and ±J L=6 at T=1, 3 seeds.
- **Falsified if:** gains come only from |C|→1 moves, or gradient variance stalls learning.
- **Cost:** ~15 min.
- **Literature:** close to PGMC (1808.09095) and Titsias & Dellaportas (1911.01373). Known in general; none found for RBM block proposals.

## Order and caveat
#1 → #2 (calibration) → #3/#4 (disorder bets) → #5 (cheapest) → #6.

For ferromagnets, gates alone already give acceptance 1. The RBM can only add value where FK fails (#3, #4), where negative results are likely (cf. 2210.11145).
