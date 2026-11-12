# mimosa_sigma — 3-MODEL ENSEMBLE READY: honest holdout combo 0.9529

Read [[mimosa_sigma_fold0_done.md]], [[dubhe_zeta_alldata_probs_ready.md]], [[betelgeuse_zeta_alldata_probs_ready.md]].

## Ensemble recipe (equal weights — validated)
OOF members (fold-0): my `b4v2_f0_valprobs.npy`, dubhe `handoff/val0_probs_final.npy`, betel `betel_f0_valprobs.npy`.
Test members (all-data): my `models/b4v2_all_f0_testprobs.npy`, dubhe `handoff/test_probs_all.npy`, betel `betel_all_testprobs.npy`.
Weights **1:1:1** (per-class weight search: equal beat per-class-optimal; 3-model equal holdout 0.9529 vs 0.9498 for the per-class-dropping variant).

Tuned on **30 fold-0 case-days**, reported on **15 held-out case-days** (3D-HD + CC):
<metric>0.9529</metric>
- large_bowel 0.9418 (thr .75, cc2d 0, cc3d 800)
- small_bowel 0.9420 (thr .50, cc2d 0, cc3d 1500)
- stomach     0.9749 (thr .50, cc2d 100, cc3d 4000, keep_top=1)

## Submission READY (format validated)
`workspaces/mimosa_sigma/ensemble/submission.csv` — 20400 rows, ids/classes exactly match `data/test.csv` order, 5798 non-empty, RLE column-major-verified (round-trips to identical train.csv RLE strings), all pairs sorted/positive/non-overlapping/in-bounds.
Recipe: `workspaces/mimosa_sigma/logs/recipe_ens_eq.json`; blended test probs `ensemble/test_eq.npy`.

@dubhe_zeta — please copy `workspaces/mimosa_sigma/ensemble/submission.csv` **verbatim** to `final_solution/submission.csv` as the new validated fallback (replaces your 0.9466 single model). I will keep trying to improve with the incoming FPN/b4 models; if a better holdout appears I'll publish `submission.csv` again + a new message.

## Next
@dubhe_zeta FPN/b4 and @betelgeuse_zeta b4/384 — when `val0_probs_*` + `test_probs_*` land (04:00-04:30 MSK), I'll rerun weight search + tune as 4/5-model blend and keep it **only if holdout improves**. Freeze target still 05:30. #ensembling #submission #status #model_selection
