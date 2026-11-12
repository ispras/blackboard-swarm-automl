# capella_kappa — final member paths (handoff to achernar)

Refs: [[achernar_delta_union9.md]], [[arcturus_kappa_verification.md]], [[capella_kappa_6model.md]]

Agreed — you own `final_solution/submission.csv`; I **stopped writing it**. My last write was the 7-member arith union (valid). My independent candidate is at `workspaces/capella_kappa/candidate_capella.csv` (9-member arith, fold-0 0.9001) as backup only.

## My FINAL member paths (val = fold-0 order; test = sample_submission order, best variant)

| member | val_probs | test_probs |
|---|---|---|
| capella_swin | `workspaces/capella_kappa/infer_swin/val_probs.npy` | `workspaces/capella_kappa/infer_swin_full/test_probs.npy` (100%) |
| capella_swin_mix | `workspaces/capella_kappa/infer_swin_mix/val_probs.npy` | `workspaces/capella_kappa/infer_swin_mix/test_probs.npy` (fold-0; mixup neutral for Swin) |
| capella_vit | `workspaces/capella_kappa/infer_vit/val_probs.npy` | `workspaces/capella_kappa/infer_vit_full/test_probs.npy` (100%) |
| capella_vit_mix | `workspaces/capella_kappa/infer_vit_mix/val_probs.npy` | `workspaces/capella_kappa/infer_vit_mix_full/test_probs.npy` (100%) |
| capella_beit_mix | `workspaces/capella_kappa/infer_beit_mix/val_probs.npy` | `workspaces/capella_kappa/infer_beit_full/test_probs.npy` (100%) |
| capella_xcit_mix | (ETA ~16:15) | (fold-0; full-data after) |

Note: I did **not** run a full-data swin_mix (mixup neutral for Swin); use `swin` (100% full) + `swin_mix` (fold-0) as two members, or just one — your call.

## Recommendation on the union
Use per-family best variants to limit correlated dilution, e.g.:
`convnext_best, v2s_best, v2m480, swin(100%), vit_mix(100%), beit_mix(100%), xcit_mix, effb4`.
Equal-weight arithmetic or geometric (both ~0.900; difference is noise). I'll have xcit ready by ~16:30 and ping you.

#communication #ensembling #submission #status
