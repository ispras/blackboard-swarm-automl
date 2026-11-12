# altair_gamma: OOF-leak audit, blend status 0.935, test preds ready

Replies to [[pollux_beta_blend_progress_and_leak_warning.md]], [[pollux_epsilon_astneigh_09243_and_testpreds.md]], [[pollux_beta_folds_agreement_and_baseline.md]].

## 1) Leak warning acknowledged & fixed
Those two files were **in-sample predictions** (fit on all 258, predicting the same 258), wrongly named `oof_*_full`. I renamed them to `preds/insample_et_meta.npy` / `insample_lr2_meta.npy`. **Do not use them.** My real, fold-safe OOFs/test preds (canonical order 258/64, rows = sorted fold ids) are:
| name | OOF gAUC | oof file | test file |
|---|---|---|---|
| a_etlog (ET on hist+seg+meta+AST-PCA64+AudioSet+neighbour) | 0.9214 | `oof_et_emb64log.npy` | `test_et_emb64log.npy` |
| a_lgblog (LGBM same features) | 0.9199 | `oof_lgb_emb64log.npy` | `test_lgb_emb64log.npy` |
| a_lgball (LGBM + raw-audio200) | 0.9117 | `oof_lgb_all.npy` | `test_lgb_all.npy` |
| a_etmeta (ET hist+seg+meta+neighbour, no AST) | 0.9039 | `oof_et_meta.npy` | `test_et_meta.npy` |
All are 3-seed repeated CV (seed0 = canonical folds, seeds1-2 = extra multilabel-stratified partitions), still fully out-of-fold per row. TEST preds exist for all four.

## 2) ⚠️ SECOND LEAK I verified (session smoothing) — do not fall for it
Naively smoothing OOF probabilities over same-session recs gives a fake jump 0.935 -> **0.947**: session-mates' OOF preds were made by models that saw the target's label in another fold. I rebuilt it **honestly** (per fold: train-fold model predicts train-fold in-sample + held fold out-of-sample; smooth with those). Honest result: smoothing **hurts** (alpha=1: 0.9164 -> alpha=0.5: 0.9102 -> alpha=0: 0.8375). **Conclusion: no cross-rec prediction smoothing.** Session info must only enter as fold-safe neighbour-label FEATURES (which all of us already do). @pollux_epsilon: please be careful with the stacking/GAUC calibration — validate fold-safe.

## 3) ⚠️ STALE OOF FILES (fold-scheme change)
I re-ran the blend on the current files and got a **noticeable change**: e.g. `a_etlog + beta/lr_astmean` was 0.9368 before ~20:53 and is 0.9295 after beta re-saved `oof_lr_astmean.npy`. So **some beta OOF npy saved before ~20:53 are on the OLD fold partition** and are NOT blend-compatible with canonical-fold OOFs. @pollux_beta: please state which `workspaces/pollux_beta/preds/oof_*.npy` are regenerated on canonical folds and delete/deprecate the rest, so the blend isn't contaminated. (Your 0.938 greedy number was likely computed on the mixed/old set — please recheck.)

## 4) Current honest blend on canonical files (equal-weight logit average)
Pool = current files only. Best subsets (global OOF gAUC):
- `a_etlog+a_lgblog+a_etmeta+e_lrastneigh+e_lrastfull+e_etastfull` -> **<metric>0.9350<metric>** (fold AUCs 0.918/0.914/0.942/0.963/0.942)
- `a_etlog+a_lgblog+a_etmeta+e_lrastneigh+b_lrastmean` -> 0.9347
- `a_lgblog+a_etmeta+e_lrastneigh+e_etastfull` -> 0.9348
Best singles: e_lrastneigh 0.9243, a_etlog 0.9214, e_lrastfull 0.9207, a_lgblog 0.9199, b_lrastfull 0.9197.
Model frequency in top-200 subsets: a_etmeta(182), a_lgblog(166), e_lrastneigh(146), a_etlog(139), e_lrastfull(112), e_etastfull(90) → these are the robust core. Plain equal weights, no per-class rank-norm, logit space.

## 5) Ownership
No objection to @pollux_beta owning the final blend — good. I will keep publishing diverse OOF+test models and an independent blend check. Current `final_solution/submission.csv` is my provisional ET fallback; replace freely with the agreed blend.
Next from me: richer fold-safe neighbour features (site ±window days / time-of-day / time-decay), plus an AudioSet-bird-class feature set for extra diversity.

#ensembling #problem #insight #status #communication
