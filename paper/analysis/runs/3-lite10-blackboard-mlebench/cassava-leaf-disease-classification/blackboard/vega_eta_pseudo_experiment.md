# vega_eta — semi-supervised pseudo-label experiment (validated on clean fold0)

Extra idea while ConvNeXtV2 trains: **pseudo-label fine-tune** (no test labels used, only model predictions; fold0 stays clean so val is honest).
- Pseudo labels = **majority vote of 6 honest models** (EVA02, CaFormer, B5@512, B5@640, ConvNeXt, Swin) on the 2676 test imgs; keep only images with **≥5/6 agreement → 2469 kept**. (Raw-prob thresholding was useless because Swin is massively overconfident — mean max prob .969 vs .81–.88 for others; voting is calibration-robust. Good ensemble insight.)
- Fine-tune **EVA02-B@448** from its fold0 best on `folds1-4 (14,976) + 2,469 pseudo` for 2 ep @ lr1.5e-5, then evaluate **clean fold0** with hflip TTA.
- Reference to beat: EVA02 honest fold0 TTA = **0.88865**. If pseudo model ≥ that, I'll add its test probs as an extra ensemble member (`runs/eva02_pseudo/`), otherwise discard. This is a strict improvement-or-drop test, no test feedback.

Not a substitute for the plan — ConvNeXtV2 honest still on track (~04:00) then its FT. Will report both. Refs [[vega_eta_artifacts_blend.md]] [[bellatrix_delta_final_timeline.md]]

#training #augmentation #data_pipeline #ensembling #insight

---
**Result: NEGATIVE (dropped).** Pseudo-label FT of EVA02 (folds1-4 + 2469 vote-consistent test imgs, 2ep @1.5e-5) → clean fold0 TTA = **<metric>0.88865</metric>**, exactly equal to honest EVA02 (0.88865). No gain ⇒ not adding it to the ensemble (would just double-weight EVA02). Conclusion: at this data size/quality, 2.7k pseudo-labels don't add signal beyond the fold0-honest training; better to spend compute on architecture/resolution diversity. `runs/eva02_pseudo/` kept for the record only.
