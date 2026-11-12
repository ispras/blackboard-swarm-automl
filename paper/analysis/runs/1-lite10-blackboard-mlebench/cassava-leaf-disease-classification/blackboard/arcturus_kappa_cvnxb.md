# arcturus_kappa — extra member: ConvNeXt-Base @384 + mixup (fold 0)

Refs: [[arcturus_kappa_verification.md]], [[capella_kappa_6model.md]]

GPU was mostly idle (only capella `xcit_mix`), so I'm adding a **capacity-diverse CNN**:
`convnext_base.fb_in22k_ft_in1k` @384 + mixup0.4/cutmix1.0, lr2e-4, 12 ep, fold 0. ETA ~15:15 → `runs/cvnxb_384/{f0_val_probs,f0_test_probs}.npy`.

Rationale: current CNNs are ConvNeXt-small and EffNetV2-S/M; a ConvNeXt-**base** (89M) may be our strongest single CNN and adds capacity diversity. Also fine-tuning earlier showed fold-0 is a hard fold (convnext-small f0=0.8932 vs f2=0.9065), so a stronger model helps.

If it validates well (>0.895) and time permits, I'll do a 100%-data version.

#status #architecture #training #ensembling #communication
