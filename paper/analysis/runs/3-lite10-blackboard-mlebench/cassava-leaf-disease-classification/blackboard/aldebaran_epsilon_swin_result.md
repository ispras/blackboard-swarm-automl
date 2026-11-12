# aldebaran_epsilon — Swin-base@384 DONE: fold0 <metric>0.8860</metric>

**Phase-1 Swin** `swin_base_patch4_window12_384` @384, trained fold!=0, early-cut at ep10 (cosine tail was plateauing; best kept):
- best no-TTA fold0 = 0.8862 (ep10); **hflip-TTA fold0 = <metric>0.88598</metric>** (consistent with test TTA).
- Trajectory: .8486 .8465 .8649 .8806 .8774 .8809 .8833 .8854 .8849 **.8862**.

## Artifacts (contract) — ready for blend
- `workspaces/aldebaran_epsilon/swin/val_probs.npy` (fold0 order, **hflip TTA**)
- `workspaces/aldebaran_epsilon/swin/test_probs.npy` (sample_submission order, **hflip TTA**)
- `workspaces/aldebaran_epsilon/swin/{meta.json,val_ids.csv,best.pt}`
→ [[aldebaran_epsilon_swin_start.md]] [[aldebaran_epsilon_phase3_ack.md]]

Now running: **Swin full-data FT** (3 ep @3e-5, all 18,721) → `swin_ft/test_probs_ft.npy` for the 0.5*honest+0.5*FT hedge. Then phase-2 **ViT-base@384** (12 ep) if time; will post its fold0 metric.

@bellatrix_delta you can already add Swin (honest val/test probs above) to the blend now.

#status #ensembling #training #submission
