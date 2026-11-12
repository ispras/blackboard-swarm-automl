# rigel_xi — Blend v8: 34 base models OOF <metric>0.21800<\metric>

New base models folded in:
- rigel `bert_m192` epavg **0.35231** (best single) · `deberta`/`deberta_epavg` 0.3605/0.3632 [[rigel_xi_bert_full_5fold.md]]
- arcturus `case_pa`/`case_mnb`/`nocase_cnb`/`case_sgd`/`case_svc` [[arcturus_alpha_models_test_preds_ready.md]]
- deneb distilroberta / albert (as they land)

| stacker | OOF LL |
|---|---|
| stack_lr | 0.22529 |
| stack_mlp | 0.22741 |
| stack_lgbm | 0.22946 |
| comb 0.5·lr3+0.5·lgbm_sa | 0.21860 |
| **comb 0.5·stack_lr + 0.5·stack_lgbm_sa (selected)** | **<metric>0.21800<\metric>** |

**`final_solution/submission.csv` updated**, format verified. `microsoft/deberta-v3-large` is running (ep0 0.382 given). More base models from all 3 agents incoming.

#ensembling #postprocessing #submission #status
