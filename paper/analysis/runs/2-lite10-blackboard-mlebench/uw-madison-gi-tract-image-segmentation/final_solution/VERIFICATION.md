# Final submission verification

File: `submission.csv` (md5 `d4f90051987dd6101aff1a397e175bfa`), frozen 3-model ensemble.

## Structural / semantic checks (all PASS)
- Exactly 20,400 rows; columns `id,class,predicted`.
- `id`/`class` order is identical to `data/test.csv`; 6,800 unique ids x 3 classes, no duplicates.
- All 3 expected classes present.
- **All 5,798 non-empty RLEs** checked: even pair count, starts >= 1, lengths >= 1, starts strictly
  increasing, no overlaps, end <= H*W (using true per-file `(H,W)`); **0 bad rows**.
- **All 5,798 masks decoded** at original `(H,W)` and verified non-empty with correct shape.
- RLE codec exactness: re-encoding 180 sampled official `train.csv` masks (266x266, 276x276,
  310x360) reproduces the original RLE strings **180/180**.

## Plausibility vs training priors
Predicted positive-slice fraction per class vs training prior:
| class | submission | train prior |
|---|---|---|
| large_bowel | 0.355 | 0.366 |
| small_bowel | 0.290 | 0.288 |
| stomach | 0.208 | 0.225 |
Mean mask area fraction per image size: 0.004-0.008 (train: ~0.016-0.022 of the full frame;
predicted masks are after 3D-CC + keep-top, consistent with the metric-optimal recipe).

## Frozen metric
Honest fold-0 holdout combo (tuned on 30 case-days, reported on 15 held-out): **0.9529**
(large_bowel 0.9418 / small_bowel 0.9420 / stomach 0.9749). Independently reproduced by
betelgeuse_zeta (0.9529). Extra members (FPN, b4) and extra morphology were rejected by holdout.
