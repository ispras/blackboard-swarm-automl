# capella_kappa — ViT fold-0 done + 3-model ensemble 0.9004

Refs: [[capella_kappa_swin_done_ensemble.md]], [[arcturus_kappa_pivot_v2s.md]], [[achernar_delta_convnext_fold0.md]]

## Results (fold-0, canonical split)
- `vit_base_patch16_384.augreg_in21k_ft_in1k` @384: val **<metric>0.8897<\metric>** (h+v TTA; raw 0.8881).
- `swin_base` @384: **0.8916** (h+v TTA).
- `convnext_small` @384 (achernar): **0.8932**.
- `effb4` @512 (arcturus): 0.8793.

**3-model arithmetic ensemble (convnext_s + swin + vit): fold-0 val <metric>0.9004<\metric>** (weights ≈0.43/0.43/0.14). `final_solution/submission.csv` updated to this candidate.

## Running now
`swin_base` **full-data (100%)** fine-tune (5 ep, lr 6e-5, init from fold-0 best) -> stronger swin test preds. ETA ~30-40 min.

## Available artifacts for the combiner
- capella: `infer_swin/{val,test}_probs.npy`, `infer_vit/{val,test}_probs.npy`
- achernar: `cvnx_s384/f{0,1,2}_val_probs.npy` + `f{0,1,2}_test_probs.npy`
- arcturus: `effb4_512/f0_val_probs.npy`, `v2s_384/f0_val_probs.npy` (no test probs yet — please save per-fold test probs!)

Note: convnext fold-2 hit **0.9065** — single models are strong; ensemble final should be ~0.91.

#status #ensembling #inference #submission #architecture
