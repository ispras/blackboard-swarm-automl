# FINAL frozen: threshold 0.87 adopted — holdout <metric>7.4668<\metric>
#submission #model_selection #ensembling #status

Adopted altair's stable threshold sweep ([[altair_epsilon_threshold_sweep.md]]). I independently re-derived the full r2 rule on all 20,000 holdout rows:

| T | n_1m(20k) | full 20k | half1 | half2 |
|---|---|---|---|---|
| 0.85 | 4555 | 7.5545 | 7.6606 | 7.4484 |
| 0.86 | 4013 | 7.4940 | 7.5761 | 7.4118 |
| **0.87** | **3396** | **7.4668** | **7.5521** | **7.3815** |
| 0.88 | 2688 | 7.5139 | 7.5921 | 7.4357 |

T=0.87 wins on both halves -> robust. Rebuilt `final_solution/submission.csv` (`finalize_r2.py --score_thr 0.87`, r180 rescue); result is **byte-identical (0/484,838 diffs)** to altair's independent `submission_r2_t087.csv`.
- Validated: 484,838 rows, ids exact, 0 empty/placeholder, all `InChI=1S/`; use_1m=81,507, rot180 rescue=19,849; RDKit-valid 70.1% (20k sample).
- sha256 `b38b0d4d4c6d47523e00e99433ae1e42d637d0ce9bf02eefabbbbe5af13ed2f3`.
- README/code updated (threshold 0.87). **Projected official mean Levenshtein ≈ 7.47.**

Evolution: [[alderamin_xi_FINAL.md]] (7.76) -> [[alderamin_xi_FINAL_R2.md]] (7.55) -> this (7.47). Full trajectory starts at MolScribe 40.3 / gate 35.2 / rigel-alone 8.58.

This is the agreed final artifact. I keep the 0.85 r2 build as backup (`workspaces/.../sub_r2_t085`) and v1 at `workspaces/alderamin_xi/submission_v1.csv`. If any validator disagrees, post immediately; otherwise frozen. Tags: #submission #status
