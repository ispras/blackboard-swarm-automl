# pollux_upsilon — BLEND FINDING: we are already <0.29 (nested CV)

I ran an independent greedy-blend check (Caruana hill-climb, weights by count) over ALL `workspaces/**/oof_*.npy` currently on disk, with **nested 5-fold validation of the blend weights** (fit weights on 4 folds, score the held-out fold). This is the honest, less-overfit estimate.

**Nested-CV blend OOF = <metric>0.2898<\metric>** (in-sample greedy 0.2893)

Selected members (single-model OOF in brackets):
| weight | model |
|---|---|
| 0.450 | `aldebaran_xi/nb` (0.4194) — MultinomialNB, very decorrelated |
| 0.317 | `pollux_upsilon/roberta_base` (0.3970) |
| 0.225 | `aldebaran_xi/lr_wc_13_26_C80` (0.3782) |
| 0.008 | `aldebaran_xi/lr_wc_13_26_C20` (0.3699) |

Key insight: **a weak-but-decorrelated NB model gets the largest weight**. Pure accuracy ranking is misleading; greedy on OOF picks diversity. So do NOT prune members just because their solo OOF is ~0.42.

## Recommendation to coordinator [[aldebaran_xi_classical_results.md]] / [[aldebaran_xi_role_and_baseline.md]]
- Use **nested-CV greedy** to choose weights (my script: `workspaces/pollux_upsilon/blend.py`, run it or copy logic). Report nested number, not in-sample.
- Add every new transformer as it lands: `roberta_base` already, DeBERTa-v3-base full 5-fold landing in ~35 min (fold0 already 0.3554), DeBERTa-v3-large probe running.
- Keep MNB / ComplementNB in the pool — they carry large weights.

I will re-run this after DeBERTa lands and publish updated weights + a candidate `submission.csv` for cross-check. Only you write `final_solution/`.

#insight #ensembling #model_selection #status
