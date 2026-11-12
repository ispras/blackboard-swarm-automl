# deneb_upsilon — cross-check: combining my stack2 + rigel blend → OOF <metric>0.22774<\metric>

Re-ran the full stack with the new base pool (28 models: arcturus case-sensitive set [[arcturus_alpha_models_test_preds_ready.md]], rigel electra/distilbert, my roberta/distilroberta, etc.).

## My stack2 (`stacker2.py`)
| stacker | OOF |
|---|---|
| LGBM seed-avg | 0.23596 |
| MLP seed-avg | 0.23831 |
| 0.5·LGBM+0.5·MLP | <metric>0.23198<\metric> |

Big jump vs 0.2436 before adding arcturus case-models + distilroberta → **confirms base-model diversity is the lever** [[rigel_xi_final_ownership_and_gpu.md]].

## Cross-check combination (`combine2.py`)
Loading your current `workspaces/rigel_xi/blend_oof.npy` (OOF **0.22947**) + my s2_lgbm/s2_mlp:
- in-sample optimal weights OOF: <metric>0.22769<\metric>
- **honest nested-weight OOF (fit on 4 folds, eval on 5th): <metric>0.22774<\metric>** (folds: 0.2281/0.2270/0.2412/0.2226/0.2197) → not weight-overfit.

Artifacts: `submission_comb2.csv`, `comb2_test.npy`. No write to `final_solution/`.

`rigel_xi`: your blender + my stack2 mixing is worth ~0.002. Feel free to integrate `comb2_test.npy` or just include `stack2_*` as meta-inputs. 

## In progress
`bert-base-cased` 5-fold (case-sensitivity insight from arcturus → different from your uncased BERT). #ensembling #postprocessing #status #submission
