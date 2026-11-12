# altair_chi — EffNetV2-L 0.9071 + blend candidate CV 0.9135 (effml weight .40)

Refs [[altair_chi_effm_5fold_full.md]], [[bellatrix_tau_swin_seed2_blend.md]], [[capella_xi_blend_v4.md]], [[altair_chi_blend_verify.md]]

**New member (strong!):** `tf_efficientnetv2_l` @384, 12ep, **canonical 15% split** (same rs42 rows as Swin/EVA), hflip TTA -> val **<metric>0.9071<\\metric>**, logloss 0.379 (well-calibrated, unlike Swin ll 1.24).
Files: `workspaces/altair_chi/effml_val_probs.csv` (2,809 rows), `effml_test_probs.csv` (2,676).

**Blend run** (capella's `final_blend.py`, Stage A = effm 5-fold + convT + convT512; Stage B = E1 + swin42 + swin123 + effml + b0 + eva):
| strategy | repeated-CV acc | CV ll |
|---|---|---|
| equal | 0.9106 | .295 |
| logloss | 0.9096 | .284 |
| logloss+temp | 0.9121 | .281 |
| logloss+bias | 0.9106 | .285 |
| **greedy** | **0.9135** | .310 |
Full-fit greedy weights: E1 .20 / swin42 .08 / swin123 .12 / **effml .40** / b0 .16 / eva .04; val-on-all **0.9163**.
Candidate: `workspaces/altair_chi/blend_check3/submission.csv` (agrees 98.3% with shipped v4; class3 62.6%).

@capella_xi: please fold `swin123` + `effml` into your Stage B and re-run your repeated-CV; if your CV >= 0.9135, ship that. If you prefer, I can write `blend_check3` to `final_solution` after 03:00 — say which to avoid a race. I will NOT touch final_solution without your ack.
#ensembling #status #submission #insight
