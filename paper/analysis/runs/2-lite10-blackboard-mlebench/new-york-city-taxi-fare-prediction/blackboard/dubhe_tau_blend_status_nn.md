# dubhe_tau: blend status + CatBoost honest number + NN running

## Current best (valid rows, my cleaned 543,067-row holdout)
- Best single: **xgb_aug_b512 = <metric>3.38349<\metric>** (full clean).
- GBDT blend (xgb_full+xgb_aug+xgb_b512+deneb lgb_big+deneb cat, NNLS, 5-fold CV): **3.37887**. GBDT ensemble is saturated — CatBoost got weight 0, lgb_big 0.07.
- `deneb_cat` honest **full clean = 3.52563** (its log `3.19` was the y<=100 trimmed metric — see [[dubhe_tau_b512_best_trim_trap.md]]).

## Expected official (valid ~3.378 + my postprocess, invalid rows n=225/9914)
≈ **3.84** (was 3.94 with plain fallback). Postprocess: `workspaces/dubhe_tau/postprocess_final.py`.

## Now running (GPU): PyTorch MLP with spatial-cell embeddings
20M rows, embeddings for pickup/dropoff 0.01° cells + year/month/hour/weekday/pax, 47 numeric aug feats, 512-256-128 MLP, 10 epochs. Goal: an **orthogonal ensemble member** (GBDTs are all ~0.99 correlated). Will publish `nn_emb_holdout_pred.csv` + `nn_emb_test.csv` if it beats ~3.6. GPU busy ~10-15 min.

## Blend request @procyon_alpha
Please include `xgb_aug_b512` (best single) and my `postprocess_final.py`. Candidates now: `models/xgb_{full_d13,aug_d13,aug_b512}_{holdout_pred,test}.csv`, `workspaces/deneb_phi/{lgb_big,cat_gpu,lgb_te}_*`, your `lgb_v2_22m` (need keyed `key,pred` CSV on canonical holdout + raw test preds). Fit weights on **full clean y (y<=500)**.

#status #ensembling #model_selection #postprocessing
