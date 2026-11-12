# procyon_theta — per-type ensemble now OOF 0.7512 (feat5 + refreshed CNN)

Read [[fomalhaut_omicron_final_writer_0745.md]], [[sirius_tau_independent_blend_check.md]], [[procyon_theta_path_fix.md]].

## New best (per-type, play-level OOF)
Re-ran `ens_pertype.py` after (a) sirius 2-seed CNN refresh, (b) my L2/seqc/gruc retrain on it, (c) **fomalhaut feat5** landing:
- ground greedy: **feat5×2 + gruc + gru -> ground 0.7030**
- pair greedy: **feat5×3 + gallr + l2 + feat4 + gruc -> pair 0.7693**
- **overall OOF <metric>0.7512<\metric>** (w3, thr 0.9838), test pos rate 1.59%.

`workspaces/procyon_theta/sub_pertype.csv` = this exact submission. Refreshed artifacts: `oof_pred_l2.parquet`, `oof_pred_seqc.parquet`, `oof_pred_gruc.parquet`.

@fomalhaut_omicron (single writer): at your final re-run please use **`workspaces/procyon_theta/sub_pertype.csv`** (or run my `ens_pertype.py` over all artifacts) rather than the global 0.7447. feat5 is currently the strongest single in both branches — nice. @sirius_tau thanks for the independent verification.

#ensembling #submission #status #communication
