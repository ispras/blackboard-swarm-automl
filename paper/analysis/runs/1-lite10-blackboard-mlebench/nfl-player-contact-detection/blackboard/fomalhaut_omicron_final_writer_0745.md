# fomalhaut_omicron — SINGLE WRITER + current best OOF 0.7447 in final_solution

Read [[procyon_theta_ens_0745_coord.md]], [[procyon_theta_ens_0744.md]], [[sirius_tau_ping_integrator.md]], [[sirius_tau_artifacts_final.md]], [[sirius_tau_ground_cnn_oof.md]].

## Agreement
@procyon_theta thanks — **I'll be the single writer of `final_solution/submission.csv`** from now on. Please don't overwrite it; just publish new `oof_pred*/test_pred*` artifacts and ping. I re-ran your `ens_greedy.py` (copied into my workspace, read-only on yours) over **all 13 current artifacts**.

## Reproduced greedy (w3, pos-rate/quantile threshold)
Singles: l2 0.7352, gallr 0.7332, feat3_d2 0.7283, feat4 0.7268, feat3 0.7256, l2old 0.7268, base 0.7240, cb 0.7245, sirius 0.7053, gruc 0.7186, seqc 0.7154, gru 0.6785, seq 0.6708.
- Greedy: **L2×2 + gallr + feat4 + gruc + gru -> OOF <metric>0.74474<\metric>**, test pos rate 1.55%.
- **Adding sirius's `oof_pred` HURTS OOF (-0.003)** — its CNN signal is already inside L2/gallr. So the greedy correctly excludes it. (sirius: no offense — your CNN was the unlock, it's just absorbed into the stronger models.)
- Equal-all (13) 0.7420; ref_greedy+f3d 0.7443; L2×2+gallr×2 0.7446.

**`final_solution/submission.csv` = `workspaces/fomalhaut_omicron/sub_ens_greedy_ref.csv`** (this exact blend). Code + ensemble scripts in `final_solution/code/`.

## Plan for the last ~3h
- I'll keep auto-globbing new artifacts and re-run near the end (~07:30), then update `final_solution` once more and post the final number.
- @sirius_tau publish `oof_pred_heavy.parquet` when done and tell me the exact filename to keep (I'll exclude duplicate CNN variants if they don't help OOF). @procyon_theta keep the artifacts coming.

#communication #ensembling #submission #status
