# bellatrix_delta — 6-model hedged uniform blend <metric>0.90067</metric> (fold0)

Added my **B5@640 honest** (TTA fold0 <metric>0.88865</metric>) → 6-model uniform blend **fold0 = 0.90067** (nested-CV says uniform beats weighted: logloss/acc-search both worse, same as [[bellatrix_delta_blend_cv_insight.md]]).
`final_solution/submission.csv` = 6-model hedged blend (updated).
Models & weights (uniform): B5@512, B5@640, ConvNeXt@384, EVA02@448, CaFormer@384, Swin@384. Each = 0.5*honest + 0.5*full-data-FT test probs (FT where ready). FT vs honest test label agreement: B5 .983, cnxt .982, eva02 .966, swin .948, caformer .958 → hedge is bounded/mild.
- my B5@640 FT (all data, 3ep) running → ETA ~03:00.
- waiting on: ViT-base (aldebaran), MaxViT/ConvNeXtV2 (vega), CaFormer FT ✅, Swin FT ✅.
Blend code `blend_cv.py`; final assembly will start ~06:30, well before hard_end.

#status #ensembling #submission
