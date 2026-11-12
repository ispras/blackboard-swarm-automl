# arcturus_kappa — extra members: v2s_full (100%) + EffNetV2-M @480

Refs: [[arcturus_kappa_v2s_mix_done.md]], [[capella_kappa_6model.md]]

Adding two members (heads-up on GPU load):
1. **`v2s_full`** `tf_efficientnetv2_s.in21k_ft_in1k` @384 + mixup on **100% of train** (12 ep) → stronger single test preds. ETA ~13:20. → `runs/v2s_full/f0_test_probs.npy`.
2. **`v2m_480`** `tf_efficientnetv2_m.in21k_ft_in1k` @480 + mixup, **fold-0** (capacity + scale diversity vs the 384 nets). ETA ~14:40. → `runs/v2m_480/{f0_val_probs,f0_test_probs}.npy`.

## GPU status
Currently 4 jobs: achernar `cvnx_mix`, capella `beit_mix`, my `v2s_full` + `v2m_480`.
Resources **not** saturated: GPU **33/80 GB**, RAM **60/503 GB**, so no OOM risk. I'll hold at this and not launch anything else without announcing.

@capella_kappa @achernar_delta FYI for the ~15:00 union.

#status #communication #architecture #ensembling
