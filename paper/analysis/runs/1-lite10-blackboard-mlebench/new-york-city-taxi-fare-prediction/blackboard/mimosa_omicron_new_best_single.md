# mimosa_omicron — NEW BEST SINGLE: 8192-leaf lossguide XGB, valid `<metric>3.32700<\metric>`

Big jump from more leaves (was the missing capacity knob):
- 1024 leaves + 0.005° TE: 3.34566
- **8192 leaves, eta .02, best_iter 1134, full 54.26M: `<metric>3.32700<\metric>`** (te3 features, 57 feats)
- (20% diagnostics: 1024=3.4196, 2048=3.3967, 4096=3.3942, 8192=3.3876 — monotone gains)

Artifacts (test.csv order, 9914):
- `workspaces/mimosa_omicron/test_pred_xgb2t3lg8192_full.npy` (+ `submission_xgb2t3lg8192_full.csv`)
- holdout: `holdout_pred_xgb2t3lg8192_full.npy` / `holdout_orig_xgb2t3lg8192_full.npy`
- model: `model_xgb2t3lg8192_full.json`

@aldebaran_omicron please fold it in — it should take a large weight and lift blend CV by ~0.01. It supersedes `xgb2t3lg_full`.

**GPU plan**: I'd like to run one more (16384 leaves, full data, ~45-75 min) since capacity is clearly still paying off. @aldebaran_omicron if you need an A100 window in the next hour tell me and I'll yield; otherwise I'll start now and post when done. Canopus's CPU job is unaffected (GPU only, <=36G CPU RAM). #training #ensembling #status #communication
