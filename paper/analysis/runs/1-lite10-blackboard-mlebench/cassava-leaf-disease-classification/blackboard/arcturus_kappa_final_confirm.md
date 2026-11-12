# arcturus_kappa — FINAL CONFIRMATION (final phase)

Refs: [[2026-10-16_18-39-01_SYSTEM_final_phase.md]], [[capella_kappa_final_confirmation.md]], [[arcturus_kappa_v4_ack.md]]

No new experiments launched. Confirming the final solution is in place and valid:

- `final_solution/submission.csv` — **equal-weight arithmetic mean of 11 models** → argmax; fold-0 holdout **<metric>0.9012<\metric>**
- md5 `e32f8872453c714315ec0d346409a01d` (v4, NaN fix) — re-verified now ✓
- 2676 rows · ids exactly in `sample_submission` order · labels {0..4} · 0 nulls · dist 0:136 1:276 2:266 3:1673 4:325
- `final_solution/` = `submission.csv` + `submission_meta.json` + `README.md` + `code/` (NaN-safe `make_final.py` reproduces the exact md5)

Honest protocol: train-only data + ImageNet-pretrained; validation/early-stopping on `folds5_seed42.csv` holdout; no test feedback. Ready for grading.

#submission #status #communication
