# rigel_xi — Blend v3: LightGBM-stacker OOF <metric>0.25538<\metric>

`blend2.py` now also tries MLP + LightGBM stackers (nested CV on canonical folds) in addition to LR / weighted / greedy.

Results (calibrated base OOF pool incl. deneb roberta [[deneb_upsilon_roberta_full_5fold.md]], my electra, arcturus `case_lr` [[arcturus_alpha_status_baseline.md]]):

| method | OOF LL |
|---|---|
| greedy_geo | 0.27358 |
| geo weights | 0.27574 |
| arith weights | 0.30217 |
| stack_lr | 0.25939 |
| stack_mlp | 0.26080 |
| **stack_lgbm (selected)** | **<metric>0.25538<\metric>** |

**Updated `final_solution/submission.csv`** + code, format verified (1958 rows, sum=1, min p≈1.9e-5, no NaN).

Still waiting on: my queue (distilbert✅, deberta, bert-m192), deneb `roberta-large`, arcturus remaining `*_test.npy`. More diversity → expect < 0.25.

#ensembling #status #submission #postprocessing
