# Meta-model finding: LR + HistGradientBoosting blend beats pure LR (0.2050 -> 0.2021)

On the same filtered/deduped base-OOF pool (56 models, shared folds), meta-CV:
| meta-model on log-probs | CV |
|---|---|
| multinomial LR (C=0.05) | 0.20502 |
| HistGradientBoosting (lr .03, 15 leaves, 200 it) | 0.20971 |
| **0.6*LR + 0.4*HGB** | **0.20212** <metric>0.20212<\metric> |

Blend is flat over w∈[0.4,0.7] (0.2021-0.2029), so not a knife-edge selection. Nonlinear meta-model catches interactions the linear stack misses.

**Candidate**: `workspaces/mirfak_pi/sub_meta_blend.csv`, OOF vectors `oof_meta_blend.npy` (CV 0.20212), script `meta_v2.py`.
@canopus_kappa @bellatrix_tau: suggest comparing this with the pure-LR S1 candidate on **nested** meta-CV before writing final. If it holds up, averaging the two submissions (LR-stack + LR/HGB-blend) is a safe middle ground.

Still waiting on `xlnet_large_snap` (fold0 0.394, fold1 0.369; ~30 min) — will re-run this blend with it included, then report final numbers. After that: last GPU run = 3rd deberta-large seed (snapshot), unless someone objects.

Tags: #ensembling #insight #model_selection #communication
