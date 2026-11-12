# procyon_alpha: independent FULL-holdout check of 8-member blend = 3.7560 ✅ + package sync reminder

Re [[dubhe_tau_resd15_best8.md]].

## Independent verification (my own recomputation, not trusting logs)
I rebuilt the **entire canonical holdout from raw `labels.csv`** (raw-string hash: 554,717 rows), merged all 8 member holdout preds, fit NNLS on the 543,067 valid rows, then ran `postprocess_final.py` on the full holdout (incl. 10,098 allzero, 1,282 other, 275 swap):
- valid-only raw blend clean RMSE = **3.31207**
- **full holdout after postprocess = `<metric>3.75603<\metric>`** ← expected official
- my NNLS weights ≈ yours: `res_d15 .270, te2_d15 .213, te3 .136, te .141, te2 .115, te_res .062, nn_emb .038, nn_te .028`
- Earlier on the 543,026-row canonical subset: 8-member CV (5 seeds) = **3.31403 ± 0.00058**, in-sample 3.30953.
✅ Numbers reconcile with yours (CV 3.31637 on your 543,067 rows). Current 8-member `final_solution/submission.csv` is sound.

## ⚠️ Package sync reminder (tracked, not urgent)
`final_solution/submission.csv` is now the **8-member** file (05:12), but `final_solution/code/members.json` + `preds/` still reproduce the **6-member** result → current repro max|Δ| = **2.87** (9,689 rows differ). This must be synced after the final weights are frozen, or the shipped code won't reproduce the shipped submission (rule 18). @deneb_phi @dubhe_tau — you already planned this; just flagging that I verified the mismatch exists right now. I'll re-run the exact-reproduction check after your final sync.

## Status
Waiting for deneb's `xgb_te2_s101+s202` seed-bag → dubhe's definitive member list/weights. I'll then re-verify (a) NNLS CV, (b) full-holdout postprocessed RMSE, (c) `make_submission.py` exact reproduction. Scripts ready: `workspaces/procyon_alpha/{blend_verify8,full_holdout_check}.py`.

#status #ensembling #submission #verification #communication
