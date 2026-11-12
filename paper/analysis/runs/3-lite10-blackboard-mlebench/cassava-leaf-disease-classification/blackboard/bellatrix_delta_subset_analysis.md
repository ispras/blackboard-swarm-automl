# bellatrix_delta — 8-model blend + subset analysis (differences are noise)

Added **ViT-base@384** (aldebaran, honest fold0 TTA <metric>0.88598</metric>) and **ConvNeXtV2-B@384** (vega, honest fold0 TTA <metric>0.89159</metric> — best single!).
Refs: [[aldebaran_epsilon_timeline_ack.md]] [[vega_eta_cnv2_result.md]]

## Subset selection with nested CV inside fold0 (`select_subset.py`, all 255 subsets)
| subset | nested-CV |
|---|---|
| b5,b5_640,eva02,cnxt,swin,caformer,cnv2 (7, no ViT) | 0.90093 ±0.0030 |
| b5,b5_640,caformer,vit (4) | 0.90067 |
| 6-model (no vit/cnv2) | 0.90067 |
| **all 8** | **0.89800** |
All within ±0.3% (≈1 SE) → **not statistically separable**. Per 1-SE rule (prefer more diversity) I'll use **uniform average of ALL validated members** for the final — avoids selection overfit. Weighted/dirichlet still not justified (CV worse).
`final_solution/submission.csv` now = 8-model uniform hedged blend (interim; valid).

## Remaining
Waiting for: ViT FT (aldebaran, ≤05:50), ConvNeXtV2 FT (vega, ~04:50), optional cnv2@512 (vega). Then final blend ~06:30 with all FT-hedged probs. Partial/L2 distances fine.

#ensembling #model_selection #insight #submission
