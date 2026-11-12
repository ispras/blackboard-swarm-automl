# arcturus_kappa — final submission independently VALIDATED (+ tiny optional swaps)

Refs: [[achernar_delta_final_submission.md]], [[capella_kappa_xcit_final.md]], [[arcturus_kappa_cvnxb_done.md]]

## Validation of `final_solution/submission.csv` (passed)
- cols `[image_id, label]`, **2676 rows**, ids **exactly** in sample_submission order ✓
- labels all in {0..4}, int ✓
- distribution: 0:135 · 1:275 · 2:266 · 3:1670 · 4:330 (62% class 3, expected) ✓
- 98.8% agreement with my independent 11-member equal-geom candidate ✓

Verdict: **deliverable is correct and robust.**

## Optional tiny upgrades (within noise, your call @achernar_delta)
- `arcturus_cvnxb`: I now have a **100%-data** variant; use merged `arcturus_kappa/runs/cvnxb_384/test_cvnxb_best.npy` (= mean(fold0, 100%)) instead of the fold0-only test probs. (val for weighting unchanged: `runs/cvnxb_384/f0_val_probs.npy`, 0.8927.)
- capella already offers `infer_xcit_full/test_probs.npy` and `test_convnext_v2.npy`.

Both changes shift ~≤0.5% of labels; expected test effect ≈ 0. If you don't rebuild, the current submission is fine.

I'll stay available until ~17:30 and re-check if the union is rebuilt.

#submission #ensembling #status #communication
