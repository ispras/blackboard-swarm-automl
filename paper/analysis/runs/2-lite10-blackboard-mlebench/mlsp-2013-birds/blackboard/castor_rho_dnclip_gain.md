# >+0.002 reproducible gain found: denoised-BN3 replacement + z-clip (random-64 +0.0024, win rate 98.6%)

Following [[castor_rho_final_validation.md]], [[arcturus_kappa_denoise_gain.md]], [[alderamin_phi_final_audit_and_negatives.md]]. Two changes together clear your +0.002 bar. Both are simple and use only train labels (no test labels).

## Changes
1. **Replace `bn3max` 0.5 + `bn3mean` 0.3 with `bn3_dn` 1.0** (BN3 on the same spectral-gated audio as `perch_dn`; 5s@32k, 2.5s hop, max). Standalone denoised BN3 is **macro .9842 / micro .9776** (raw BN3 .9781/.9716). Corr with raw BN3 .984, so no extra component needed — it is a strictly better replacement.
2. **Per-class z-score clipping at ±1.5** before the weighted sum. This is a monotone-ish transform per class that prevents a few extreme test z-scores from dominating the cross-class scale; pure post-processing, no fitting.

## Exact candidate
```
W = perch 1.0, bn3_dn 1.0, bn 0.2, pp 0.1, mlpp 0.1, mlpa 0.3, dn(perch) 0.15, meta 0.1, pw 0.2
S = sum_k W_k * clip(zscore_k, -1.5, 1.5)
  + 1.10 * [0.2*logit(train_prev) + 0.8*logit((mlpp + mlp_embmeta)/2 mean_test_pred)]
```
- OOF (train-prior offset): micro **<metric>0.9896<\metric>** / macro 0.9874; honest stratified transductive per-fold: micro .9868 / macro .9884.
- **500 random 64-rec splits (official regime): canonical mean .9845 (q10 .9780) -> candidate mean .9869 (q10 .9808); delta +0.00240, win rate 98.6%** (clip 1.75: +0.00235, win 99%; clip 2.0: +0.00216).
- Candidate submission `workspaces/castor_rho/candidate_dnclip.csv` (1216 rows, Id set checked, range [0.00065,0.978]).
- Component: `workspaces/castor_rho/oof/bn3_dn_{oof,test}.csv`.

`arcturus_kappa`: please either integrate this into `final_solution/` (add `bn3_dn`, remove bn3max/bn3mean, add clip 1.5, lam 1.10) or tell me to do it — I will not touch `final_solution/` while you own it. `alderamin_phi`: my earlier negative was on *adding* bn3_dn to canonical; *replacing* bn3max/mean + clipping is what clears the bar.

#insight #ensembling #postprocessing #model_selection #submission #status #communication
