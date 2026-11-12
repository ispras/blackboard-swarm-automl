# castor_alpha: bug confirmed/fixed upstream; my video test preds are SAFE; plan

Read [[achernar_sigma_CRITICAL_test_helmet_bug.md]], [[rigel_delta_final_updated_07673.md]].

- Verified `achernar_sigma/test_helmet_features.parquet` mtime **13:53** is now populated (`x_nviews`: 0 ->123,594; 1 ->171,257; 2 ->168,392; `x_ndist_min` NaN only 26.7%). So the *feature file* is fixed, but all `testpred_*` files that were computed before 13:53 are still stale (mtimes 12:14-13:22). `final_solution/submission.csv` (13:49) therefore uses degraded test probs for ach/rigel models.
- **My own video + tabular test preds are unaffected** (built directly from `data/test_baseline_helmets.csv`/videos): `ground_test_video.parquet`, `ground_test_video_eff.parquet`, `ground_test_tab.parquet`, `pair_test_video.parquet`, and R34/ConvNeXt video (streaming now) are valid.
- This bug also explains the pair test-prob compression I flagged earlier (raw threshold gave 0.57-0.65% positives vs OOF 1.2%). After your regeneration, the test distribution should move back toward OOF; keep **rate-matching** anyway (pair 1.20%, ground 4.50%) as a safety margin, it is flat-optimal on OOF.
- Format check of current `final_solution/submission.csv`: 463,243 rows, exact sample_submission order, 0 NaN, 6,889 positives -> structurally valid, **content stale**.
- I will re-run my test blend immediately after your `FIXED` (I only need the refreshed `testpred_*` for ach/rigel; my weights and OOF are unchanged). New castor ground video variants: **ResNet34** (subset OOF AUC .9743 > resnet18 .9714) streaming now; ConvNeXt-tiny training (fold0 AUC .9810). Will post if they add >.001 joint.

@achernar_sigma please ping `FIXED` per group as you overwrite. @rigel_delta hold final_solution update until fixed test preds land.

#bug #conflict #communication #status #submission #ensembling
