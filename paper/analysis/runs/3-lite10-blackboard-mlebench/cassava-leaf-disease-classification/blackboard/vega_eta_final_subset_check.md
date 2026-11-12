# vega_eta — last offline check: current uniform-8 is the right choice (no change)

Using the FINAL TTA val probs (all 8 members), nested 5-fold CV inside fold0:
- **uniform 8 = 0.89880** (matches the frozen ensemble's reported fold0)
- drop-one: drop ViT 0.89987, drop Swin 0.89880, drop cnv2 0.89853, drop b5/b5_640/caformer 0.89800, drop cnxt/eva02 0.89773.
- adding ConvNeXtV2@512 as a 9th: 0.89826 (worse), so correctly excluded.
All deltas ≤ +0.001 (≈4 imgs, well inside 1 SE ≈0.005). **No subset is statistically better than the current uniform-8 ⇒ keep the frozen file unchanged.** The frozen `final_solution/submission.csv` (06:13, fold0 0.89880) is final; I re-verified it with 0/2676 label mismatches. [[vega_eta_FINAL_verification_v2.md]] [[bellatrix_delta_FINAL_submission.md]]

#ensembling #model_selection #insight #submission #status
