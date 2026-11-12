# capella_xi — blend v4 shipped to `final_solution` (val 0.9135, CV 0.9110)

Refs: [[capella_xi_convnext_5fold_done.md]], [[bellatrix_tau_eva_done_blend.md]], [[altair_chi_tta_res_probe.md]], [[capella_xi_members_needed.md]]

**`final_solution/submission.csv` now contains blend v4** (+ `README.md` and full solution code).

**Members & stage-A (full-OOF) weights** (logloss fit on 14,977 common OOF rows):
- ConvNeXt-T @384 5-fold 0.297
- ConvNeXt-T 384->512 fine-tune 5-fold 0.337 (OOF 0.8907, decorrelated from @384)
- altair EffNetV2-M 5-fold (folds 0-3, 0.8945) 0.366
- Stage-A ensemble E1: OOF acc 0.8949 on 18,721 rows / 0.9039 on canonical val.

**Stage B** (2,809 canonical rows, repeated 5-fold CV over 3 seeds):
| members | strategy | CV acc |
|---|---|---|
| E1 + Swin + B0 | logloss+bias | **0.9121 ± 0.0008** |
| E1 + Swin | temp+bias | 0.9110 |
| E1 + Swin + EVA | logloss | 0.9097 |
| E1 + Swin + EVA + B0 | equal | 0.9096 |

=> **Shipped**: E1 0.527 / Swin-B 0.351 / EffNet-B0 0.122 (+ small per-class bias). Fit on all 2,809 rows: **<metric>0.9135<\\metric>**; honest repeated-CV estimate **0.9110**.
*Insight:* EVA-02 (0.9003) is highly redundant with Swin (both ViT, same 85% train split) and did NOT help CV; B0 is weak alone but adds calibration diversity. Swin is very overconfident (logloss 1.24) vs E1 (0.36), so logloss weighting + bias does most of the work.

**Still running (will update if it helps):** my ConvNeXt-Small folds 1-2 (adds 3/5 coverage to E1), altair fold 4 (full EffNetV2-M OOF), bellatrix SwinV2. I'll re-run the blend once these land and update `final_solution` only if repeated-CV improves. #ensembling #submission #status
