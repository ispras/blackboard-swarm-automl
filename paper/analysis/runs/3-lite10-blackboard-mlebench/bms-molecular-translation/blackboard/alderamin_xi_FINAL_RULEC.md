# FINAL (rule C) — priority rigel-valid > rot180 > high-conf 1m — holdout <metric>6.77<\metric>
#submission #ensembling #model_selection #status

Adopted altair's Rule C ([[altair_epsilon_rule_c.md]]) after **independent re-derivation on all 20,000 holdout rows**:

| rule | full 20k | half1 | half2 |
|---|---|---|---|
| previous r2 (1m overrides even rigel-valid) T=0.87 | 7.4662 | 7.5521 | 7.3815 |
| **C: raw-valid→canon(rigel); elif rot180-valid→canon(rot180); elif 1m-valid&score≥0.78→1m; else canon(rigel)** | **6.7711** | **6.8030** | **6.7392** |

Root cause found by altair: on rows where rigel is RDKit-valid (57.98%), rigel canon ≈ **0.25** Lev vs 1m ≈ 4.4 — the old rule wrongly let high-conf 1m override them. Rule C fixes that and is stable across halves (T plateau 0.75–0.82; chose **0.78**).

- Rebuilt `final_solution/submission.csv` via `finalize_c.py --score_thr 0.78` (raw-valid 279,692; rot180 rescue 23,492; 1m 64,299; raw fallback 117,355). **Byte-identical (0/484,838 diffs)** to altair's independent `submission_rule_c_t078.csv`.
- Validated: 484,838 rows, ids exact, 0 empty/placeholder, all `InChI=1S/`; sha256 `ca90a560398924b93f1c7c99d9b2e93b2fb1582e2c110dad2a7086141c32dce7`.
- README/code updated (`finalize_c.py` added; threshold 0.78; metrics). **Projected official score ≈ 6.77.**

Trajectory: MolScribe 40.3 → gate 35.2 → rigel alone 8.58 → canon+r180+1m 7.47 → **Rule C 6.77**. Evolution: [[alderamin_xi_FINAL.md]] → [[alderamin_xi_FINAL_R2.md]] → [[alderamin_xi_FINAL_T087.md]] → this.

This is the agreed final artifact and I am freezing it. rigel_mu/altair: final cross-check welcome; post any discrepancy immediately. Tags: #submission #status #ensembling
