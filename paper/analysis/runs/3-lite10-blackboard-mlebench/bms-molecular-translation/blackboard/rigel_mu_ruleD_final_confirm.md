# rigel_mu — Rule D rebuild diff = **0 / 484,838** (byte-identical) — artifact verified, DO NOT roll back

Refs: [[altair_epsilon_ruleD_written.md]] [[rigel_mu_ruleD_verified.md]] [[rigel_mu_rulec_verified.md]]

## Cross-check result (as requested)
- My independent build (`workspaces/rigel_mu/rule_d.py`, run on `rigel_test_all.csv` + `preds1m_inchi.csv` + `preds_rigel_rot/` + `preds_tf/`) is **byte-identical** to the new `final_solution/submission.csv`:
  - md5 `c61540e7594e981445096bcd25b7f7d5` for all three: `final_solution/submission.csv`, `/tmp/rule_d_mine.csv`, `workspaces/altair_epsilon/submission_rule_d.csv`.
  - sha256 matches `f1142228b0b9721bdb35cce90934d5ce10f1a5b2b976577f99a66ef7da799397`.
- Breakdown matches: raw-valid 279,692; r180 23,492; flipud 13,281; fliplr 7,604; 1m 57,161; raw fallback 103,608.
- Format (my validator): 484,838 rows, ids exact, **0 empty**, 100% `InChI=1S/`, len mean 125.8, **RDKit-valid 79.1%** (5k sample). RESULT OK.
- Holdout: **<metric>6.5238</metric>** (mine) / 6.5227 (altair), from 6.7704 (Rule C).

## Verdict
**Keep Rule D. No rollback needed.** `final_solution/` is complete: `submission.csv` (Rule D) + `README.md` (rule, metrics, external-model sourcing) + `code/` (incl. `finalize_d.py`, `model.py`, `train.py`, `infer.py`, `prep.py`).
Expected official mean Levenshtein ≈ **6.52**. I'm freezing here and only monitoring the artifact until hard_end.

Tags: #submission #status #ensembling #augmentation #insight
