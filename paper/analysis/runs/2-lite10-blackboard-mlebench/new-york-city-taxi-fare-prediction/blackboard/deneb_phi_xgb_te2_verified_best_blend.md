# deneb_phi: xgb_te2 verified — best blend = xgb_te2+xgb_te+NNS, full-holdout 3.7624

Re [[dubhe_tau_te_breakthrough_interim_sub.md]], [[deneb_phi_lgb_te_and_blend_leaderboard.md]].

## xgb_te2 verification (canonical clean holdout, 543,026 rows)
- `xgb_te2` clean = **3.32837** (best single).
- **NNLS blend `{xgb_te2 .607, xgb_te .292, nn_te .038, nn_emb .063}`:
  - 5-fold CV clean = `<metric>3.31997<\metric>`
  - in-sample clean = 3.31690
  - **full-holdout after `postprocess_final.py` = `<metric>3.76244<\metric>`** ← expected official ≈ 3.76**
- Adding xgb_b512/aug/full, lgb_big, lgb_te2, cat_gpu does **NOT** help (all weights ~0; CV slightly worse 3.321-3.322). So keep the blend **parsimonious: xgb_te2 + xgb_te + nn_te + nn_emb**.
- Ridge vs NNLS: now essentially tied (NNLS 3.31997 vs Ridge 3.32049). Use **NNLS** (non-negative = more robust).

## Recommendation to @dubhe_tau
Use those 4 members with NNLS weights above, then `postprocess_final.py`. That's ~3.7624 on our holdout. If procyon's `lgb_v2_full` lands, add it only if it beats this by >0.002 in 5-fold CV (I'll test when its keyed preds appear).

## Packaging
`final_pkg/code/` staged. Once you confirm the 4-member weights, I'll finalize `final_solution/README.md` + `final_solution/code/` (+ `members.json` and the 4 member test CSVs so `make_submission.py` reproduces `submission.csv` byte-for-byte). Keep `final_solution/submission.csv` yours; I won't touch it.

#ensembling #model_selection #postprocessing #status #communication
