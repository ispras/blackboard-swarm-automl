# capella_kappa — wrap-up (my members + validation)

Refs: [[achernar_delta_final_v3.md]], [[arcturus_kappa_v3_validated.md]], [[capella_kappa_swinmix_full.md]]

Swarm solution is converged and validated: `final_solution/submission.csv` = **equal-weight arithmetic mean of 11 models**, fold-0 holdout **<metric>0.9012<\metric>** (best single ~0.895). Format verified (2676 rows, sample order, labels 0-4, no nulls); `final_solution/` has submission.csv + submission_meta.json + README.md + code/.

## My (capella_kappa) contributions
| model | fold-0 val |
|---|---|
| `swin_base` @384 (100%) | 0.8916 |
| `vit_base_patch16_384` + mixup (100%) | **<metric>0.8943<\metric>** (best single member overall) |
| `xcit_medium_24` + mixup (100%) | 0.8884 |
| `beit_base_384` + mixup (100%) | 0.8884 |
| `swin_base` + mixup (100%) | 0.8889 |
| `vit_base_patch16_384` (100%) | 0.8897 |

## Contributions beyond models
- Caught the 90/10 vs 5-fold split bug -> authored canonical `workspaces/shared/folds5_seed42.csv` (now used by everyone).
- Authored `workspaces/capella_kappa/{train.py,infer.py,ensemble.py}` (multi-scale/hv TTA, mixup/cutmix, blend combiner) reused across the swarm.
- Independently confirmed the equal-weight/geometric-vs-learned conclusion and produced matching candidate submissions (98.7-99.6% agreement with final).
- Provided the `best-variant` (100%-data) members and flagged correlated-model dilution.

All training used only `data/train_images` + ImageNet-pretrained weights; validation strictly on the shared fold-0 holdout. Standing by / monitoring the board — no further heavy jobs launched.

#status #submission #ensembling #communication #insight
