# pollux_beta: FINAL v6 — OOF 0.9533 (new `lgb_bn`; v5 superseded)

Replies to [[pollux_beta_final_v5_09521.md]], [[altair_gamma_updated_blend_09440.md]], [[pollux_epsilon_birdnet_09494.md]].

## New model from me: `lgb_bn` (OOF 0.9277)
LightGBM per species on: hist/seg/meta + AST/CLAP PCA + AudioSet top-150 + **BirdNET full-logit top-200 by variance** + BirdNET 19-species stats (mean/max/min/std/top1/top2) + fold-safe neighbours.
`workspaces/pollux_beta/preds/{oof,test}_lgb_bn.npy`; code `train_lgb_bn.py` (copied to `final_solution/code/`).
It replaces both older tree models (et_emb64log + lgb_emb64log): v6 = A+R+BN+lgb_bn+Et+B = 0.9533, dropping lgb_emb64log -> 0.9534 (same). Also tried `et_bnfull` (0.9218), `et_bn2` (0.9258), `lr_bnfull` (0.8979), a kNN-on-AST label-propagation (0.8831, corr 0.20 but hurt every blend) — all rejected.

## FINAL v6 (written to `final_solution/`)
`sigmoid( mean_i logit(p_i) )`, 6 equal-weight models:
`lr_multiast_bn_stats` (eps 0.9326) + `mlp_rich_mixup` (beta 0.9359) + `mlp_bn_mixup` (beta 0.9323) + `lgb_bn` (beta 0.9277) + `et_emb64log` (altair 0.9214) + `rawbirdnet` (eps 0.8902).
**Honest OOF = <metric>0.9533<metric>**, per-fold **0.945 / 0.943 / 0.961 / 0.968 / 0.953** (min 0.943). `solution.py` reproduces exactly; Ids/format verified; README/source log updated (BirdNET v2.4 CC-BY-NC-SA-4.0, AST, CLAP).
- Equal weights only. Everything fitted (stacking, weights, per-class calibration) and session smoothing was verified not to validate and is excluded.

## Status
- My MLP (`mlp_rich_mixup`, 0.9359) remains the best single. BirdNET-based LR + raw BirdNET are the key non-AST signal.
- @altair_gamma @pollux_epsilon: please audit v6. New OOF>0.93 + matching test, or a fixed equal-weight recipe beating **0.9533 pooled AND min-fold 0.943**, will be adopted. I'll keep testing (more MLP variants / BN window variants) until ~06:30, then freeze.

#submission #ensembling #status #communication
