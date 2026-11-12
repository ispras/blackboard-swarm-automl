# procyon_theta — IMPORTANT path fix for single writer + new L2 retrain

Read [[fomalhaut_omicron_final_writer_0745.md]], [[sirius_tau_independent_blend_check.md]].

## ⚠️ Path correction for the final write
The **per-type 0.7468** submission is **`workspaces/procyon_theta/sub_pertype.csv`** (produced by `workspaces/procyon_theta/ens_pertype.py`). It is **NOT** `sub_ens_greedy.csv` (that one is the global 0.7447). @sirius_tau referenced the wrong file. Please have @fomalhaut_omicron copy **`sub_pertype.csv`** (or re-run `ens_pertype.py`) as the final, consistent with your B plan.

Confirmation it uses the global rank scale + single global threshold (per [[sirius_tau_independent_blend_check.md]]): ranks are computed over all 4.26M rows first, then per-type weighted, concatenated, one global w5 threshold 0.98357 in rank space.

## I am not writing final_solution anymore
Respecting [[fomalhaut_omicron_final_writer_0745.md]]: **fomalhaut_omicron is the only writer.** I'll only publish artifacts.

## In progress (will ping)
sirius updated the CNN in place (2-seed, +ground AUC). I'm **retraining my L2 and the seq CNN/GRU with the updated CNN** then re-running per-type; if OOF > 0.7468 I'll publish `oof_pred_l2.parquet` + `sub_pertype.csv` and ping @fomalhaut_omicron to re-glob before 07:30.

#communication #ensembling #submission #status
