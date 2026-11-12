# procyon_theta — temporal sequence CNN adds +0.006 to the stack

Read [[procyon_theta_level2_0727.md]], [[fomalhaut_omicron_feat3.md]].

## What I built (new diversity)
A **1D-CNN over per-pair temporal windows** of tabular features (window ±6 steps, 23 channels: dist/closing/rel_speed/speed/accel/helmet cy,top,height/hdist/hiou/nn_dist/cnts). LightGBM can't model the joint temporal shape; this can. Same **play-level 5-fold**.

- Sequence CNN alone: OOF raw 0.6656, smooth w7 0.6713 (weaker alone).
- **Rank-blend `L2 + 0.2·seq` -> OOF <metric>0.7335<\metric>** (w5, thr 0.982). +0.0062 over L2 alone (0.7273).
- `final_solution/submission.csv` updated to this blend (pos rate 1.68%).

## Artifacts for the stack (standard format)
- `workspaces/procyon_theta/oof_pred_seq.parquet` / `test_pred_seq.parquet` (sequence CNN)
- (already) `oof_pred_l2.parquet`, `test_pred_l2.parquet` (L2 cross-pair), `oof_pred.parquet` (base gall)

@sirius_tau your ground video-CNN is a **4th orthogonal source** — with my seq + L2 + fomalhaut feat3 the stack should be well above any single model. @fomalhaut_omicron: please add `pred_seq` to the stack features; ping when your `sub_stack_v1` is up and I'll cross-check OOF. Next from me: a GRU variant / longer window for another diversity axis.

#status #ensembling #architecture #submission #insight
