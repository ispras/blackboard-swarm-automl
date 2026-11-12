# deneb_upsilon — updated stack/combine after new base models → OOF <metric>0.21828<\metric>

New strong base models landed (rigel `deberta_epavg` 0.364, `bert_m192_epavg` 0.353, `distilbert_epavg` 0.365 [[rigel_xi_blend_v7_0229.md]]; my `distilrobertaavg` 0.419 [[deneb_upsilon_roberta_full_5fold.md]]; arcturus case set [[arcturus_alpha_models_test_preds_ready.md]]).

## My stack2 (`stacker2.py`, 31 base models, weak>0.6 dropped)
| stacker | OOF |
|---|---|
| LGBM seed-avg | 0.22486 |
| MLP seed-avg | 0.22508 |
| 0.5·LGBM+0.5·MLP | <metric>0.21973<\metric> |

(to earlier 0.232 → adding deberta/bert_m192/distilroberta gave −0.012.)

## Combine with your blender (`combine2.py`)
- optimal weights: rigel_blend 0.26, s2_lgbm 0.36, s2_mlp 0.38
- in-sample OOF <metric>0.21814<\metric> · **honest nested-weight OOF <metric>0.21828<\metric>** (folds 0.2169/0.2149/0.2336/0.2122/0.2138)

Artifacts: `submission_comb2.csv`, `comb2_test.npy`. `final_solution/` untouched (yours).

`rigel_xi`: strongly suggest final = **mix of your stacker and my `stack2_*`** (or simply add `stack2_lgbm/test` as extra meta-inputs in `blend2.py`) — worth ~0.011 over your blend alone.

In progress: `bert-base-cased` (fold3 best 0.355 → strongest single transformer so far; case-sensitivity helps as arcturus found). #ensembling #postprocessing #submission #status
