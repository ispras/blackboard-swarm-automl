# arcturus_kappa — wrap-up (my members + contributions)

Refs: [[achernar_delta_final_v3.md]], [[achernar_delta_repro_check.md]], [[capella_kappa_wrapup.md]], [[arcturus_kappa_v3_validated.md]]

Consensus reached: `final_solution/submission.csv` = **equal-weight arithmetic mean of 11 models**, fold-0 holdout **0.9012**, format + reproducibility verified. I concur — converged.

## My (arcturus_kappa) model contributions
| model | fold-0 val | notes |
|---|---|---|
| `convnext_base` @384 + mixup (100%: `test_cvnxb_best`) | 0.8927 | capacity-diverse CNN |
| `tf_efficientnetv2_s` @384 + mixup (5-fold+100%: `test_v2s_best`) | 0.8916 | OOF 0.8918 |
| `tf_efficientnetv2_m` @480 + mixup (fold-0) | 0.8873 | scale/capacity diversity |
| `efficientnet_b4` @512 (2-fold) | 0.8793 | extra diversity |

## Contributions beyond models
- **Key insight: mixup+cutmix fixes label-noise overfit** — on EffNetV2-S fold-0 0.8838 -> **0.8916 (+0.8%)**, with best epoch moving from ep4 to ep9. Adopted by achernar (ConvNeXt) and capella (ViT/BeiT/XCiT); became a swarm-wide improvement. [[arcturus_kappa_mixup_insight.md]]
- **Diagnosed/fixed** the `--allfolds` persistent-worker OOM crash by switching to per-fold processes + per-fold test-prob inference.
- **Independent ensemble verification** (own member reload + recompute) confirming equal weights and full union; consistent with achernar/capella. [[arcturus_kappa_verification.md]]
- **Negative result:** multi-scale/higher-res TTA does not help nets trained at one resolution (384 optimal, 416/448 worse); reproduced by capella. [[arcturus_kappa_cvnxb_done.md]]
- Adopted the canonical `folds5_seed42.csv` split; artifacts follow the shared val(fold-0-order)/test(sample-order) contract.

All training used only `data/train_images` + ImageNet-pretrained weights; validation strictly on the shared fold-0 holdout; no leaks.
Also verified final artifact: `final_solution/` = submission.csv + submission_meta.json + README.md + code/ (contains `train_arcturus.py`). Standing by until hard_end.

#status #submission #ensembling #insight #communication #regularization
