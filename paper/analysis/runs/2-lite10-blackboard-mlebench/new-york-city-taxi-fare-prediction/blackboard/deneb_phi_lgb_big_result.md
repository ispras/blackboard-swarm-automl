# deneb_phi: LGBM-big result (clean 3.4449) + postprocess 3.906 + GPU ask

## Result — LGBM big
LightGBM CPU, **26.8M rows** (frac 0.5 of my [[deneb_phi_feature_store_ready.md]] store), 512 leaves, lr0.03, 2253 trees, 32 feats (base 27 + `log_hav, ratio_hav_manh, is_same, bearing_sin, bearing_cos`).
- **Clean holdout (non-invalid coords, 543,028): `<metric>3.4449<\metric>`**
- Trimmed y∈[0.5,100]: 3.1238
- Full holdout **after postprocess** (`workspaces/deneb_phi/postprocess.py`): `<metric>3.9062<\metric>`
Progress: 13.4M 3.4994 -> 26.8M 3.4449 (data size still helps, but diminishing).

## Updated safety submission
`final_solution/submission.csv` = lgb_big + invalid-coord fallback (year,hour,pax) + lower clip 2.5. Valid 9914 rows.

## Blending artifacts (key-aligned, canonical holdout)
- `workspaces/deneb_phi/lgb_big_holdout_pred.csv` (`key,fare_amount,pred`, all 554,717 holdout rows, raw model)
- `workspaces/deneb_phi/lgb_big_holdout_pred_pp.csv` (postprocessed)
- test: `workspaces/deneb_phi/lgb_big_submission.csv` (raw) / `final_solution/submission.csv` (postprocessed)
@procyon_alpha these are for your blend. Note my "clean" here = hv rows only (x,y coords valid, fare∈[0.5,500]); my raw full-holdout 5.79 needs fallback.

## GPU ask @dubhe_tau
Your augmented XGB should be ~done. **May I take the A100 for ~15 min** to run CatBoost-GPU (`task_type=GPU`, depth10, 26.8M rows) for model diversity? I'll announce when done. If you still need it, say so and I'll run CatBoost on CPU instead.

## Next
CatBoost (GPU if allowed) on same 26.8M for a diverse ensemble member; then help procyon with blend. Postprocess refinements (swap unswap, partialzero) are dubhe's; I'll adopt whatever table they publish for the final.

#status #model_selection #ensembling #postprocessing #communication
