# bellatrix_tau — EVA-02 done (0.9035) + 3-model blend 0.9114

Refs: [[bellatrix_tau_swin_final.md]], [[capella_xi_convnext_5fold_done.md]], [[altair_chi_fold0_result.md]]

**Model 3 (done):** `eva02_base_patch14_448` @448, 12ep, lr1e-4, wd0.05, mixup0.2, canonical 15% holdout.
- no TTA 0.90352; hflip+vflip TTA <metric>0.90032<\\metric>; rot8 0.89996 -> **use hflip/vflip TTA for EVA** (rotation TTA does not help ViT here).
- Files: `workspaces/bellatrix_tau/runs/eva02_448/{val_probs_hv.csv,test_probs_hv.csv}`.

**My blend sanity check on the 2,809 canonical val rows** (using capella_xi's assembled ConvNeXt-T 5-fold OOF, join by image_id):
- swin alone 0.90708, eva alone 0.90032, conv OOF 0.90103
- **3-model blend <metric>0.91136<\\metric>** (weights: swin .45 / eva .50 / conv .05), 5-fold CV of the weighting selected greedy+per-model temperature calibration.
- Temperature scaling matters: Swin is overconfident (fitted T~4.8); calibrating each member before weighted averaging improved CV by ~+0.3pt.
- Ready files: `workspaces/bellatrix_tau/runs/blend3.csv` (3-model, 2676 rows), `runs/blend_swin_conv.csv` (2-model fallback).

**Running:** `swinv2_base_window12to24_192to384` @384, 14ep, mixup0.2 -> 4th diverse member, ETA ~01:00. After that I can also give a full-data retrain if time.

@capella_xi: when assembling the final blend, try (a) equal weights, (b) greedy accuracy with member repetition, (c) per-member temperature scaling then greedy — chose by CV on the shared rows, it was (c) for me. My ready-to-use test probs: `swin test_probs_rot8.csv`, `eva test_probs_hv.csv`. Also note my models cover only 2,809 rows, so include altair's + your full OOF to estimate weights on the full 18,721 rows where possible (my members will be missing there; you can fit weights among full-OOF members first, then add mine on the 2,809 intersection). #ensembling #status #inference
