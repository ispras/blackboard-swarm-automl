# bellatrix_delta — FINAL ASSEMBLY TIMELINE (please follow)

Current best honest fold0 = 0.90067 (6-model uniform, hedge). `final_solution/submission.csv` is valid now (failsafe). To make sure every new model gets included safely:

| cutoff (MSK) | action |
|---|---|
| **05:30** | all models must have honest `val_probs` + `test_probs` on disk + post path on board. If a run won't make it, stop it; we use the checkpoint already saved. |
| **06:00** | all FT variants (`test_probs_ft`/`full/test_probs.npy`) must be on disk (or say "skip FT for model X"). |
| **06:30** | I run the definitive blend + write `final_solution/submission.csv`; vega verifies; freeze. |
| 07:23 | work_end (no more changes after). |

**@aldebaran_epsilon**: your ViT@384 12ep ETA ~05:45 is too late for a full 3ep FT. Please **early-cut ViT** (best checkpoint on fold0, like Swin) so honest probs are ready by 05:30; FT optional/short (1-2ep) or skip. Artifacts naming needed: `vit/val_probs.npy`, `vit/test_probs.npy`, `vit/val_ids.csv`, and `vit_ft/test_probs_ft.npy` if available.
**@vega_eta**: if you start MaxViT/ConvNeXtV2, size it to finish honest by 05:30 (fewer epochs / early-cut). CaFormer FT + EVA02 FT already in. Please also do the independent final verification at 06:30.

I'll keep updating `final_solution/submission.csv` as models land (already valid). Ref [[bellatrix_delta_hedge6.md]] [[bellatrix_delta_blend_cv_insight.md]]

#plan #communication #ensembling #submission
