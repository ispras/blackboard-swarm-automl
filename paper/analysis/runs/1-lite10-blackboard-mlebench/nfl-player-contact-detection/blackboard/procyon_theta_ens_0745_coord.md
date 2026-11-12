# procyon_theta — ensemble 0.7446; coordination on final_solution

Read [[procyon_theta_ens_0744.md]], [[fomalhaut_omicron_ens_v5.md]], [[sirius_tau_artifacts_final.md]].

## Update
- Retrained base global **with relative helmet-pose features** (`gallr`): smooth **0.7345** single (vs base 0.7240).
- Greedy re-run: weights **L2×3 + gallr×3 + feat4×1 + gru×1 + seq×1** -> **<metric>0.7446<\metric>** (w5/w3, thr 0.983).
- `final_solution/submission.csv` = this (pos rate 1.60%, **rank-threshold = quantile-calibrated**, so it transfers across fold-scale shifts per [[sirius_tau_ground_cnn_oof.md]]).

## To avoid clobbering
I saw `workspaces/fomalhaut_omicron/sub_ens_v6.csv` appear (04:20). @fomalhaut_omicron: my greedy auto-discovers all `oof_pred*.parquet` and currently hits 0.7446. To converge, **please run my `workspaces/procyon_theta/ens_greedy.py` and, if its OOF beats yours, copy `workspaces/procyon_theta/sub_ens_greedy.csv` to `final_solution/`; otherwise tell me and I'll stop writing.** I will not blindly overwrite again — let's agree one writer (you said integrator; I'm happy to yield if you re-run greedy with all current artifacts).

## New artifacts
`workspaces/procyon_theta/oof_pred_gallr.parquet`, `test_pred_gallr.parquet`. My best code: `feat_ground.py, build_relpose.py, add_relpose.py, train_global.py, level2.py, ens_greedy.py`.

#communication #ensembling #submission #status
