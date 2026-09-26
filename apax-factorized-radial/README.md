> **Note:** this study was carried out with Claude (Anthropic's AI assistant) under close human oversight.
> None of the conclusions here are final.

# Factorized species-pair radial functions for GMNN (apax)

Replacing GMNN's dense (119 × 119 × n_radial × n_basis) species-pair radial coefficient table with a low-rank
CP factorisation over per-element embeddings, tested on MAD-1.6 dimers/trimers and ethanol.

- `FACTORIZED_RADIAL.md` — full write-up: motivation, literature, method, cost/precision analysis, all experiments.
- `CRITIQUE.md` — critical review, open concerns, prioritised next steps.
- `EXCLUDED_DATA.md` — what isn't in this repo (model checkpoints, training data, logs) and how to regenerate it.
- Scripts: `mad_dimtri_subset.py` (data subsets), `eval_multi.py`, `eval_fbins.py`, `eval_mad5k.py`,
  `eval_wd_s1.py`, `core_rank.py` (evaluation/analysis), `run_*.sh` (launchers), `config_*.yaml` (every run).
