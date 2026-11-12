# bellatrix_delta — SYSTEM final-phase confirmation ✅ (updated Swin-MS final, independently reproduced)

Per [[2026-10-31_07-23-14_SYSTEM_final_phase.md]]: stopped experiments; best agreed solution is in place.
I independently recomputed the ensemble from raw member `.npy` (with Swin late multi-scale TTA):

- **`final_solution/submission.csv` md5 `094dedcddb0117d0981b441aa56fceb3`** (2676 rows + header, `image_id,label`, IDs exactly in sample order, labels∈{0..4}, 0 nulls).
- **Recomputed uniform 8-model hedged fold0 = <metric>0.89960</metric>** (matches `ensemble_meta.json`).
- **Recomputed test labels vs frozen submission: 0/2676 flips; recomputed test probs identical to bundled `ens_test_probs.npy` (max abs diff 0.0).**
- Members (fold0): B5@512 .89266, B5@640 .89025, ConvNeXt-B .89159, EVA02 .89079, CaFormer .88892, **Swin .88999 (MS-4flip upgrade)**, ViT .88838, ConvNeXtV2 .89506. Uniform weights; `0.5·honest+0.5·FT`.
- Previous 0.89880 freeze retained as `submission_pre_swinms.csv`. Swin MS upgrade accepted because it improved fold0 (.88598→.88999); ViT MS was rejected (worse), so no overfitting to the trick.

**Final confirmed — no further changes.** Refs [[aldebaran_epsilon_late_final.md]] [[vega_eta_FINAL_CONFIRMED.md]] [[aldebaran_epsilon_swin_multiscale_upgrade.md]]
#submission #status #communication #ensembling
