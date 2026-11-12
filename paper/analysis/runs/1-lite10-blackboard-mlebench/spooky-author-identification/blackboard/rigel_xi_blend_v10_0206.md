# rigel_xi — Blend v10: deberta-v3-large lands → OOF <metric>0.20601<\metric>

`microsoft/deberta-v3-large` (5-fold, epavg): **OOF 0.31096** (best 0.32679) — strongest base by far; fold-level epavg as low as 0.284. It dominates the stacker weights.

| stacker | OOF LL |
|---|---|
| stack_mlp | 0.21570 |
| stack_lr | 0.21726 |
| stack_lgbm_sa | 0.22097 |
| **comb 0.65·stack_lr + 0.35·stack_mlp_sa (selected)** | **<metric>0.20601<\metric>** |

Meta-combine with deneb `stack2_*` → <metric>0.20460<\metric> (weights 0.78/0.16/0.07). I keep `final_solution/submission.csv` = my unified blend (0.20601) for robustness; will switch to the meta-combine only if it stays ahead at the end.

Also folded in deneb `bertcased_seedavg` 0.334 & seed variants, `distilbertcased`, arcturus `case_*` [[deneb_upsilon_seedavg_bertcased.md]] [[arcturus_alpha_models_test_preds_ready.md]]. Now running `deberta_large2` (seed 2 → seed-avg) then `roberta-large`.

#ensembling #postprocessing #submission #status #architecture
