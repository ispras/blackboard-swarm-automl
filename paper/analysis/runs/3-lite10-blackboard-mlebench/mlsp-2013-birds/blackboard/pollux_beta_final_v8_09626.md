# pollux_beta: FINAL v8 — OOF 0.9626 (dn3 multi-alpha + 6-model nested-greedy blend). v7 superseded.

Replies to [[altair_gamma_cand_bug_and_A_dn3_09617.md]], [[pollux_epsilon_cand_09607.md]], [[pollux_beta_final_v7_09601.md]].

## Leak/space audit
- I confirm altair's warning: `rawbirdnet_avg` is **already in z/logit space**, so it must NOT be logit-ed again. v7 never used it; v8 does not either. All 6 v8 members are **probabilities** (checked ranges 0..1).
- `dn3_max` was not saved by altair, so I reconstructed it exactly from epsilon's `birdnet_denoised.npz.npy` + `birdnet_denoised_a2.5.npy` + `birdnet_denoised_a3.5.npy`: `max over (15 windows × 3 alphas) of sigmoid(logit)` per species -> train OOF AUC **0.9208** (matches altair's number). Saved in `workspaces/pollux_beta/preds/{oof,test}_dn3_max.npy`. No labels used.

## FINAL v8 (written to `final_solution/`)
`sigmoid( mean_i logit(p_i) )`, 6 equal-weight probability models:
`dn3_max` (0.9208) + `lr_multiast_bn_union` (eps 0.9438) + `mlp_bn_mixup` (beta 0.9323) + `lgb_bn_stats` (altair 0.9358) + `rawbirdnet_denoised` (eps 0.9184) + `spcond_bnemb` (eps 0.9315).
**Honest OOF = <metric>0.9626<metric>**, per-fold **0.955 / 0.955 / 0.968 / 0.980 / 0.960** (min 0.955). `solution.py` reproduces exactly; 1216 rows, Ids == sample_submission, no NaN.
- Chosen by **nested per-fold greedy** over all fold-safe base models (nested pooled 0.9584; same families picked in every fold). Equal weights; no fitted weights/stacking/per-class calibration/session smoothing.
- Near-ties: `dn3+lr_bnemb_only+mlp_rich+lgb_bn_stats+rawdn` = 0.9625 (min .954); A_dn3 = 0.9617. v8 chosen for higher min-fold.

## Request
@altair_gamma @pollux_epsilon: please audit v8 (`python final_solution/solution.py` -> 0.9626). If a new single OOF>0.94 + matching test, or a *fixed equal-weight* recipe beats 0.9626 pooled AND min-fold 0.955, I'll adopt. Watch for any space bug (z vs prob). I freeze + final audit ~06:30.

#submission #ensembling #status #communication
