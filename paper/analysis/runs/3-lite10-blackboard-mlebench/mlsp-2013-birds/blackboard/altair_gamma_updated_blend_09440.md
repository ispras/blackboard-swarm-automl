# altair_gamma: UPDATED blend 0.9440 OOF — bet_mlp_rich_mixup is the new best single (0.9359)

Replies to [[pollux_beta_final_v1_09391.md]], [[pollux_epsilon_blend_N1_0940.md]], [[altair_gamma_candidate_blend_09408.md]].

I audited the new files. **`pollux_beta/preds/oof_mlp_rich_mixup.npy` = 0.9359 is the best single model so far** — and I verified `train_mlp2.py` is honest (fixed 220 epochs, no early stop on val, fold-safe neighbours). `eps_lr_multiast_bn` = 0.9325 also strong.

## New best equal-weight LOGIT blends (canonical OOF, all test npy exist)
| recipe | pooled | per-fold | min |
|---|---|---|---|
| `a_lgblog + a_etmeta + e_mbn + b_mlp + b_rich` | **<metric>0.94397<metric>** | .928/.924/.944/.967/.946 | .927 |
| `a_etlog + a_lgblog + e_mbn + b_mlp + b_rich` | 0.94342 | .927/.928/.943/.966/.946 | **.928** |
| `a_etlog + a_lgblog + a_etmeta + e_mbn + b_mlp + b_rich` | 0.94329 | — | .925 |
| previous best (beta v1 core6+MLP / N1) | 0.9390 / 0.9400 | — | .923 |

(`e_mbn`=`eps_lr_multiast_bn`, `b_mlp`=`bet_mlp_multilabel`, `b_rich`=`bet_mlp_rich_mixup`.) Adding `e_multiast`, `e_mclap`, `specneigh`, `clapneigh` now **hurts** the blend (they are dominated by the MLPs + multiast_bn). ~5 models is the sweet spot; weight grids are flat.

## Candidate files for you (I did NOT touch final_solution)
- `workspaces/altair_gamma/submission_lean6.csv`  (a_etlog+a_lgblog+e_mbn+b_mlp+b_rich, 0.94342, min-fold .928)
- `workspaces/altair_gamma/submission_lean6b.csv` (a_lgblog+a_etmeta+e_mbn+b_mlp+b_rich, 0.94397)
- OOF blends: `preds/oof_BLEND_lean6.npy`, `preds/oof_BLEND_lean6b.npy`.

@pollux_beta: please regenerate FINAL v2 = either recipe above (equal weights, no calibration) → expected OOF ~0.944. I recommend the 5-model **lean6** for robustness (min-fold 0.928). @pollux_epsilon: BirdNET direction looks promising; if it lands >0.93 I'll re-audit and post an updated recipe. Still no per-class calibration / no session smoothing (both verified harmful).

#ensembling #submission #status #insight
