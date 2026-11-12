# NEW BEST CANDIDATE: gate+TTA submission ready — holdout <metric>33.66<\metric> (512-row paired set)

Refs: [[altair_epsilon_tta2_plan.md]] [[alderamin_xi_tta_rot_priority.md]] [[rigel_mu_val_curve.md]] [[altair_epsilon_rescue_done.md]]

## TTA rescue finished (1m rot90/180/270 on all 50,694 both-invalid test ids)
- valid rates on test: rot90 36.2%, rot270 36.9%, rot180 30.7%; **>=1 valid rotation: 29,969 (59.1%)** — matches the 57.2% holdout estimate.
- Rule: highest-score valid among {90,180,270} -> overwrite only those both-invalid rows; everything else keeps the gate (1m-valid / 680k-valid / formula).
- **`workspaces/altair_epsilon/submission_final.csv`** (484,838 rows, 100% coverage, 0.48% empty strings, **89.9% RDKit-valid InChI**).

## Holdout estimate (my 512-row paired set, same protocol)
- gate alone: **35.38**
- both-invalid subset (53 rows) baseline 99.23 -> TTA-mixed **82.55**
- => **final ≈ <metric>33.66<\metric>** on the 512-row holdout (alderamin's 208-row set implies ~32.9; your call on which to quote).
- Individual TTA configs on those 53: rot90 only -> overall 34.39; rot90+270 -> 33.93; **rot90+180+270 -> 33.66** (best).

## Remaining plan
- `rigel_mu` keeps training to ~15:45. Then on the **20,725 both-invalid rows still without a valid TTA prediction**, test `TTA-if-valid else rigel` vs `rigel` vs `TTA`. Expected further **-1 to -3** if rigel reaches ~40-50 (`[[rigel_mu_val_curve.md]]`).
- **alderamin**: please validate `submission_final.csv` (independent) and be ready to write the final `final_solution/submission.csv`; I'll hand the last merge as `submission_final_rigel.csv` after 16:15. Note current `final_solution/submission.csv` (12:09) differs from my gate on ~9,836 rows — please reconcile which fallback it used.
- I'm CPU-only now; no GPU jobs from me until rigel finishes.

Tags: #submission #augmentation #ensembling #postprocessing #status #insight
