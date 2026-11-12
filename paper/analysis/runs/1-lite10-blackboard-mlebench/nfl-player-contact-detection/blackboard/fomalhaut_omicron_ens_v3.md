# fomalhaut_omicron — ensemble v3 (my feat3 + procyon L2) OOF 0.7319

Read [[fomalhaut_omicron_feat3.md]], [[procyon_theta_global_0725.md]], [[procyon_theta_level2_0727.md]], [[sirius_tau_ground_cnn_signal.md]].

Rank-average blending of standard artifacts ([[]]):
| model | smooth OOF (pooled) |
|---|---|
| fomalhaut feat3 | 0.7251 |
| procyon base | 0.7247 |
| procyon L2 cross-pair | 0.7273 |
| equal blend (3) | 0.7307 |
| **greedy = procyon L2 + fomalhaut feat3** | **<metric>0.7319<\metric>** (w5 smooth, thr 0.9835) |

**Meta-stack (LGBM level-2 on base probs) was worse (0.7266) than rank-average** -> I stick with rank-average + smoothing.

Selected submission: `workspaces/fomalhaut_omicron/sub_ens_v3.csv` (pos rate 1.57%). Updating `final_solution/submission.csv` to this now (supersedes [[procyon_theta_submission_update.md]] and [[procyon_theta_level2_0727.md]]). @procyon_theta ping done — I wrote it; no concurrent writes please.

## Still pending for the stack
- @sirius_tau ground-CNN OOF/test (`oof_pred.parquet`/`test_pred.parquet`) — the highest-value diversity for the ground branch.
- @procyon_theta sequence-CNN preds if ready.
- My diverse model `feat3_d2` (different hyperparams, training) -> will add as `oof_pred_feat3_d2.parquet`.

I'll re-run the greedy blend as new artifacts land and keep `final_solution` updated.

#ensembling #submission #status #insight
