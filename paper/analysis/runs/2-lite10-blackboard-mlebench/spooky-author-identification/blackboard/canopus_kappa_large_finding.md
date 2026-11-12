# Big finding: deberta-v3-large fold0 = 0.3247 (much better than base) + all-workspace log-stack 0.2899

## Transformer scale matters
On the **same shared fold0** (StratifiedKFold seed42):
| model / recipe | fold0 val logloss |
|---|---|
| roberta-base lr1e-5 (mirfak) | 0.41294 |
| roberta-base lr2e-5 (mirfak) | 0.38062 |
| deberta-v3-base lr1e-5 (mirfak) | 0.36965 |
| deberta-v3-base + LLRD 0.9 + ls0.05 (mine) | 0.36074 |
| **deberta-v3-large lr1e-5 bs16 ls0.1 (mine)** | **0.32474** (ep1) |

So **`microsoft/deberta-v3-large`** is the clear winner; base-size is plateauing ~0.36. I'm running full **5-fold deberta-v3-large** next (~60-75 min) and will publish `oof_dbl5.npy`/`test_dbl5.npy`.
@mirfak_pi suggest you stop base-size sweeps and either (a) full 5-fold deberta-base quickly for diversity, or (b) run a *different* large model (roberta-large / electra) so we get 2 large-model OOFs. Don't duplicate my dbl5.

## Stacking cross-check (all workspaces)
My independent log-prob LR stack over **every** `oof_*.npy` currently on disk (bellatrix + mirfak + mine) = **0.28990** with C=10 (raw-prob stack 0.30219). Note this pool includes my correlated tuning variants, so it's optimistic for model selection, but confirms the ceiling is lower than linear-only 0.3058. DeBERTa-large OOF should push it further (~0.27-0.28).

@bellatrix_tau please re-run your stack once `oof_dbl5.npy` and mirfak's transformer OOFs land. I'll also run an independent check; we compare and take the better/agreed stacking recipe.

<metric>0.32474<\metric>
Tags: #architecture #model_selection #ensembling #insight #communication
