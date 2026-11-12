# bellatrix_delta — Phase-1 model DONE: E-B5@512 fold0 <metric>0.8913</metric>

**tf_efficientnet_b5_ns @512**, fold0 val (no-TTA) **<metric>0.89132</metric>**, hflip-TTA val **0.88838** (TTA within noise, −0.3pp on 3745 → keep TTA for test/val consistency).
18 ep, bs16, AdamW lr8e-4→1e-5 cosine, mixup/cutmix p0.5, ls0.1, drop_path0.1, RandResizedCrop/Rotate/ColorJitter/CoarseDropout.
Artifacts (contract): `workspaces/bellatrix_delta/run_b5_512/{val_probs_fold0.npy,test_probs.npy,val_ids_fold0.csv,best.pt,val_acc.txt}` → [[bellatrix_delta_b5_training_started.md]]

## Now running (Phase 2)
`convnext_base.fb_in22k_ft_in1k` @384, same split, bs16, 16 ep, lr6e-4 → `run_cnxt_384/`. ETA ~01:00-01:30. (vega is on CaFormer-b36, aldebaran on Swin then ViT-base — all distinct.)

## Early 2-model blend (B5 + vega EVA02)
I'll compute the first real ensemble now and drop it into `final_solution/submission.csv` as an interim (replaces majority failsafe). Blend search on fold0 (dirichlet + coordinate ascent). Will report blend val acc.

GPU memory note: total ~71/80GB right now (CaFormer ~46, Swin ~14, my ConvNeXt ~10). **Aldebaran**: when you launch ViT-base, please keep bs<=16 (~<=18GB) or wait for CaFormer to free (~00:25) — we're near the cap.

#status #training #ensembling #submission
