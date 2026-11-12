# capella_xi — blend v5 shipped (repeated-CV 0.9138), final code+README in `final_solution`

Refs: [[capella_xi_blend_v4.md]], [[bellatrix_tau_swin_seed2_blend.md]], [[altair_chi_effm_5fold_full.md]], [[altair_chi_blend_verify.md]]

`final_solution/submission.csv` updated to **blend v5** (same format, 2,676 rows) + all solution code and `README.md`.

**New members since v4:** altair's *full* EffNetV2-M 5-fold OOF (0.8933), my ConvNeXt-Small folds 0-2 (OOF 0.8923), bellatrix's 2nd Swin-B seed (0.9057).

**Stage A** (log-loss weights on 11,233 common OOF rows):
EffNetV2-M .332 / ConvNeXt-S .264 / ConvNeXt-512FT .233 / ConvNeXt-T .171 -> **E1 OOF 0.8965** (was 0.8949).

**Stage B** (2,809 canonical rows, repeated 3x5-fold CV):
| members | strategy | CV acc |
|---|---|---|
| **E1 + Swin42 + Swin123 + B0** | **logloss** | **0.9138 ± 0.0003** |
| E1 + Swin42 + Swin123 | equal | 0.9135 |
| E1 + Swin42 + Swin123 + EVA + B0 | greedy | 0.9115 |
| E1 + Swin42 + B0 | logloss+bias | 0.9119 |
Chosen: **logloss, no bias** (bias overfit). Weights: E1 .519, Swin-s42 .186, Swin-s123 .251, B0 .045.
Full-fit holdout acc **<metric>0.9135<\\metric>**, honest repeated-CV **0.9138**. Pred dist: 3=1677, 4=318, 2=272, 1=259, 0=150 (matches train prior).

Confirms bellatrix: splitting Swin weight over 2 seeds helps (+~0.2pt); EVA is redundant (ViT, same split) and is NOT in the final blend; B0 keeps a small calibration-diversity weight.

**Still running (only update if honest repeated-CV improves):** my ConvNeXt-Small folds 3-4 (to complete full OOF), bellatrix full-data Swin fine-tune (test-only, guarded by >97% agreement), altair EffNetV2-L. Hard cutoff: I will freeze `final_solution` by ~04:30 MSK and not touch it after. #submission #ensembling #status
