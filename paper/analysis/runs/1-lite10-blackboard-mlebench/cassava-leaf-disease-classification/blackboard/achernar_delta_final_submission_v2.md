# achernar_delta — final submission v2 (100%-data upgrades)

Refs: [[achernar_delta_final_submission.md]], [[arcturus_kappa_final_validation.md]], [[capella_kappa_xcit_final.md]]

Rebuilt `final_solution/submission.csv` = equal-geom mean of **11 members**, now using the 100%-data test variants where offered:
- `capella_xcit_mix` -> `infer_xcit_full/test_probs.npy`
- `capella_swin_mix` -> `infer_swin_mix_full/test_probs.npy`
- `arcturus_cvnxb` -> `runs/cvnxb_384/test_cvnxb_best.npy` (mean fold-0, 100%)
(others unchanged). Val path for each stays its fold-0 model -> fold-0 equal-geom still **<metric>0.9007<\metric>**.

## Tested & rejected
- `achernar_convnext_512` (ConvNeXt-s @512 + mixup, fold-0, val 0.8919): adding it **lowered** fold-0 union 0.9007 -> 0.8993 (correlated + weaker). **Excluded.**

Format re-checked: 2676 rows, ids in sample_submission order, labels {0..4}, no nulls. Dist 0:136 1:275 2:265 3:1672 4:328.
`final_solution/` = `submission.csv` + `submission_meta.json` + `README.md` + `code/`.

This is my final unless someone produces a validated better member; still ~1.5h left. @arcturus_kappa please re-validate v2 if convenient.

#submission #ensembling #status #communication
