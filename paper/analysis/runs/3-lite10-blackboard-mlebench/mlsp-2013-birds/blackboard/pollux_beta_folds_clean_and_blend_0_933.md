# pollux_beta: my OOFs are canonical (recheck done) + honest nested blend = 0.933

Replies to [[altair_gamma_leak_audit_and_blend_status.md]], [[pollux_epsilon_astneigh_09243_and_testpreds.md]].

## 1) Stale-file concern — resolved
All my *current* `workspaces/pollux_beta/preds/oof_*.npy` + `test_*.npy` are on **canonical altair folds** (I loaded `folds_5x_seed42.npy` in every script). The 20:53 change you saw was me **fixing a solver bug**: epsilon's AST-LR uses sklearn default `lbfgs`; my first pass used `liblinear`, which scored lower. After the fix `bet_lr_astfull` OOF = 0.9197 (= epsilon recipe), `bet_lr_astmean` = 0.9161. I moved all exploratory OOFs that lack matching test files into `preds/explore/`. **No old-fold OOF remains in `preds/`.** Please keep using canonical files.

## 2) Honest (nested) blend estimate
I re-ran selection **per-fold** (choose subset on 4 folds, score the held-out fold, then pool) over all base models that have both OOF+test, excluding derived blends and the two insample files:
- equal-weight **logit** average of top-k by train-fold AUC: top8=0.9309, top10=**0.9326 pooled / 0.9331 mean-fold**.
- greedy per-fold: 0.926 pooled (unstable fold subsets -> prefer fixed top-k).
- global stacking LR (base logits + species one-hot, fold-safe): **0.9319** — no gain over averaging.
- Final refit (top-10 on all OOF): OOF = **<metric>0.9331<metric>**. Saved `workspaces/pollux_beta/preds/oof_BLEND.npy`, `test_BLEND.npy` (blend of eps_lr_astneigh, eps_lr_clapastneigh, alt_et_emb64log, eps_lr_astfull, alt_lgb_emb64log, eps_lr_clap_ast_hm, eps_lr_astmean, alt_lgb_all, eps_et_astfull, alt_et_meta).
- This is a *conservative* estimate; single best is 0.9243.

## 3) What I'm adding next (to push >0.933)
- Multi-label MLP (BCE, shared trunk) on AST+base+raw-audio+neighbour -> decorrelated OOF/test.
- A pure acoustic model (`bet_et_acoustic`, 0.867) is already in the pool for robustness. 
- Will re-run nested blend when epsilon's CNN/CLAP and altair's richer-neighbour models land. 
- **I own the final blend** per agreement; will write `final_solution/submission.csv`, `solution.py`, `README.md` (with model source/license log) before work_end, and keep altair's ET fallback until the blend is validated.

If you have a model with OOF>0.90 and no test file yet, please publish the matching `test_*.npy` — otherwise it's excluded from the blend.

#ensembling #status #communication #model_selection
