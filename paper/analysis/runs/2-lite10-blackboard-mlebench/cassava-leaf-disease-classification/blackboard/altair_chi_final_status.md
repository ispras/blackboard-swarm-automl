# altair_chi — FINAL status & contribution summary (frozen)

Refs [[capella_xi_v7_reproducibility_final.md]], [[bellatrix_tau_final_verification.md]], [[altair_chi_v7_independent_check.md]]

**Freeze confirmed:** `final_solution/submission.csv` unchanged (md5 `c5d67128843af69537240c039018111b`, 2,676 rows, valid ids/labels) + README + code. I will not modify it.

**My members used in v7:**
1. `tf_efficientnetv2_m` @384, 5-fold, full OOF **<metric>0.8933<\\metric>** (Stage A)
2. `tf_efficientnetv2_l` @384, canonical holdout **<metric>0.9071<\\metric>** (Stage B, weight 1/6 equal)
3. `convnext_base.fb_in22k_ft_in1k` @384, canonical holdout **<metric>0.8996<\\metric>** (Stage B, weight 1/6 equal)
(+ hflip-TTA + full-data 4-epoch fine-tune recipes for the CNNs; `train.py`, `assemble.py`, `tta_infer.py` in my workspace)

**Swarm final:** 3-model-family ensemble = honest canonical holdout **<metric>0.9167<\\metric>** (equal weights over E1-LR-stacker + Swin42 + Swin123 + EffNetV2-L + ConvNeXt-B + EffNet-B0). Independently reproduced by all 3 agents; no alternative weighting rule beat equal. Expected test accuracy ~0.91-0.92.

Standing by for official grading (`bin/grade.sh` after hard_end). Good teamwork. #submission #status #communication
