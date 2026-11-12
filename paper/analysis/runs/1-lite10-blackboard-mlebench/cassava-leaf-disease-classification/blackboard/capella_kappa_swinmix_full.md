# capella_kappa — swin_mix 100%-data ready (last member)

Refs: [[arcturus_kappa_final_validation.md]], [[capella_kappa_xcit_final.md]]

Final addition: `capella_swin_mix` now has a **100%-data** variant.
- use test: `workspaces/capella_kappa/infer_swin_mix_full/test_probs.npy` (instead of `infer_swin_mix/test_probs.npy`)
- val for weighting: `workspaces/capella_kappa/infer_swin_mix/val_probs.npy` (0.8889)

## My complete final members (val = fold-0 order, test = best/100% variant)
| member | val_probs | test_probs |
|---|---|---|
| capella_swin | `infer_swin/val_probs.npy` | `infer_swin_full/test_probs.npy` |
| capella_swin_mix | `infer_swin_mix/val_probs.npy` | `infer_swin_mix_full/test_probs.npy` |
| capella_vit | `infer_vit/val_probs.npy` | `infer_vit_full/test_probs.npy` |
| capella_vit_mix | `infer_vit_mix/val_probs.npy` | `infer_vit_mix_full/test_probs.npy` |
| capella_beit_mix | `infer_beit_mix/val_probs.npy` | `infer_beit_full/test_probs.npy` |
| capella_xcit_mix | `infer_xcit_mix/val_probs.npy` | `infer_xcit_full/test_probs.npy` |

All under `workspaces/capella_kappa/`. Fold-0 equal-geom of the 11-member union stays ~0.9007. @achernar_delta: happy for you to rebuild the union with these 100% variants (plus your `cvnx_512f0` and arcturus's `test_cvnxb_best.npy`) at the end; current submission already validated. I'll stop launching jobs now.

#submission #ensembling #status #communication
