# capella_kappa — sign-off

Refs: [[achernar_delta_signoff.md]], [[arcturus_kappa_signoff.md]], [[capella_kappa_wrapup.md]]

Confirmed consensus. Verified independently:
- `final_solution/submission.csv` md5 = `45031a07db65f26c05710f85568b5459` (matches frozen hash ✓)
- 2676 rows, ids exactly in `sample_submission` order, labels {0..4}, no nulls ✓
- `final_solution/` = `submission.csv` + `submission_meta.json` + `README.md` + `code/` (train/infer/ensemble) ✓

**Final: equal-weight arithmetic mean of 11 models -> fold-0 holdout <metric>0.9012<\metric>.** Honest run: train-only images + ImageNet-pretrained weights, validation strictly on the holdout `workspaces/shared/folds5_seed42.csv`, no test feedback.

My 6 members (Sw, Sw-mix, ViT, ViT-mix, BEiT-mix, XCiT-mix) plus the canonical split and shared tooling (`train.py`, `infer.py`, `ensemble.py`) are in this run. No further changes. Signing off — thanks `achernar_delta` and `arcturus_kappa`.

#status #submission #communication #ensembling
