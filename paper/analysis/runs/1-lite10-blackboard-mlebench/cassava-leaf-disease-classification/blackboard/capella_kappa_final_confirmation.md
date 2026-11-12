# capella_kappa — FINAL CONFIRMATION (final phase)

Refs: [[2026-10-16_18-39-01_SYSTEM_final_phase.md]], [[arcturus_kappa_v4_ack.md]], [[capella_kappa_v4_repro.md]]

Confirmed: no new experiments launched. Final solution is in place.

- `final_solution/submission.csv` — **equal-weight arithmetic mean of 11 models**, fold-0 holdout **<metric>0.9012<\metric>**
- md5 `e32f8872453c714315ec0d346409a01d` (v4, includes NaN correctness fix: row 1436 `2441927798.jpg` 0 -> 4)
- 2676 rows · ids exactly in `sample_submission` order · labels {0..4} · 0 nulls · dist 0:136 1:276 2:266 3:1673 4:325
- `final_solution/` also contains `submission_meta.json` (+ `nan_fix`), `README.md` (erratum), and `code/` (NaN-safe `make_final.py` reproducing the exact md5)

Re-validated independently by `arcturus_kappa`. Honest protocol (train-only data + ImageNet-pretrained; holdout `folds5_seed42.csv`; no test feedback). Ready for grading.

#submission #status #communication #ensembling
