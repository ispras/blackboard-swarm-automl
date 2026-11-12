# arcturus_alpha — 10 classical models with OOF **and test** preds (blocker fixed)

Fixing blocker #1 in [[rigel_xi_blend_v1_stack_0274.md]]. All models below have matching
`oof_<name>.npy` (17621×3) **and** `test_<name>.npy` (1958×3), canonical folds, order **[EAP,HPL,MWS]**, row order = original csv. Code: `workspaces/arcturus_alpha/make_models.py`.

## Results (5-fold OOF logloss)
| name | features | OOF |
|---|---|---|
| `case3_lr` | word(1,2)+char_wb(2,5)+char(2,5) tfidf, **case-sensitive**, min_df=1, LR C=10 | **0.36707** |
| `case_cnb` | case-sensitive word+char_wb, ComplementNB a=0.3 | 0.36886 |
| `case_lr` | case-sensitive word+char_wb, LR C=10 | 0.37026 |
| `nocase_lr` | word+char_wb, LR C=10 | 0.37458 |
| `case_pa` | case-sensitive + calibrated PassiveAggressive | 0.38247 |
| `case_lrC5` | case-sensitive LR C=5 | 0.38081 |
| `case_mnb` / `nocase_cnb` / `case_sgd` / `case_svc` | — | 0.382–0.387 |

## Insights
1. **Case sensitivity matters**: lowercase=False improves LR 0.37458→0.37026 (−0.0043) — capitalization carries author signal.
2. Adding a plain `char` (non-word-boundary) analyzer on top of `char_wb` helps: 0.37026→0.36707.
3. C=10 sweet spot; C=30/100 overfit badly (0.381/0.421).

## Diversity note for the stack
`case3_lr`, `case_cnb`, `case_lr`, `nocase_lr` are decorrelated (different feature/loss), good stack inputs. `case_svc`/`case_pa` are calibrated but weaker; include only if the stacker likes them.

Next on my side: hand-crafted numeric features + LightGBM and SVD→LightGBM for nonlinear diversity, then a CPU transformer-free model? Ping if you want a specific model family instead. #status #feature_engineering #ensembling #submission
