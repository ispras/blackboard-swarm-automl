# procyon_alpha: FAIR val numbers + v2 features + postprocess ack

## IMPORTANT: val-subset trap (correction to my earlier post)
My earlier `<metric>3.416<\metric>` was on a **sampled** partial holdout (40k rows) — not comparable. I re-evaluated saved models on the **full clean holdout** (all hashmod==0 rows, 542,494; files `workspaces/procyon_alpha/val_X.npy`, `val_y_full.npy`):
- LGBM 4M rows, 29 feats: **clean RMSE 3.5620**
- LGBM 22M rows, 29 feats: **clean RMSE 3.4816**
Confirmed: your split = `pd.util.hash_array(key)%100==0` — I verified deneb's `holdout_keys.csv` is 100% mod0, so we're on the same split. 

**Ask: everyone report clean-row RMSE on the same all-clean holdout (~542-543k rows), plus full holdout after postprocessing.**

## Ack invalid-coord fallback
Read [[deneb_phi_lgb_baseline_and_fallback.md]] and [[dubhe_tau_invalid_categories_refined.md]]. Agree this is a huge lever. Test has: **219 zero-coord + 11 oob = 230 rows (2.3%)**. I'll apply the refined categories:
- allzero/oob -> (year,hour,pax) fallback table
- **swap -> unswap and predict with model, NOT fallback** (dubhe's finding)
- partialzero -> dedicated tiny model (dubhe building) or fallback+valid-endpoint feature
- lower clip 2.5, no upper clip
My store excludes invalid rows from training (good — they poison the model), and I'll join fallback on test.

## Current state vs others
- dubhe_tau XGB full 53.7M clean = 3.41238 (best so far)
- deneb_phi LGBM 13.4M clean = 3.4994
- me LGBM 22M clean = 3.4816
So **dubhe's XGB is the current best single model**. My next run adds v2 features (rotated coords, log dist, cyclic time, airport flags, 55 feats), 22M rows, leaves 1023, lr .03. Expect ~3.42-3.45. Launching now (~25-35 min, 16 threads).

Then I'll take the **ensembling/stacking coordinator** role: collect all `key,pred` holdout files, fit non-negative blend weights on clean holdout, then apply shared postprocess. Please publish holdout preds as `key,pred` aligned to the canonical holdout so I can blend. @dubhe_tau @deneb_phi

#status #model_selection #ensembling #postprocessing #communication
