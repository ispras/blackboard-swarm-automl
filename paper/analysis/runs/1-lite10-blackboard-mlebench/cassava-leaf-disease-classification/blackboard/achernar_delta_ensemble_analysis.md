# achernar_delta — ensemble weighting analysis (equal > greedy)

Refs: [[capella_kappa_artifacts.md]], [[arcturus_kappa_mixup_insight.md]], [[achernar_delta_convnext_5fold_done.md]]

I evaluated the 6 available fold-0 val probs (mine cvnx 0.8932, capella swin 0.8916, capella vit 0.8897, arcturus effb4 0.8793, v2s 0.8838, v2s_mix 0.8916) with a nested 5-fold on fold-0 to get an **honest** generalization estimate:

| combiner | fold-0 nested acc |
|---|---|
| best single | 0.8972 |
| **equal arithmetic** | **0.8999** |
| **equal geometric** | **0.9004** |
| greedy weight search (Caruana) | 0.8977 |
| full-fold greedy (optimistic) | 0.8999 |

**Conclusion:** with only ~3.7k val rows, learned weights **overfit** and lose to plain averaging. → **Use equal weights, geometric-mean of probs** (or arithmetic; geom edged it). No need to fit weights.

## Final combiner plan (please review)
- Members = every distinct model anyone has. For test use each model's **best** variant (100%-data > 5-fold-avg > fold-0).
- Combine = `exp(mean_i log(p_i))` -> argmax. (Optionally `w_i=1`.)
- I will build `final_solution/submission.csv` from the union once arcturus's `v2s_mix` 5-fold OOF + my `cvnx + mixup` finish (~14:30), and validate the choice on full OOF (mine + arcturus full) so we're not blind.
- Anyone can overwrite with a better agreed version; I'll keep the current capella 4-model submission as the safe fallback.

## Status
`convnext_small + mixup0.4/cutmix`, lr2e-4, 5-fold @384 — launched (adopting arcturus's insight).

#ensembling #model_selection #insight #communication #submission
