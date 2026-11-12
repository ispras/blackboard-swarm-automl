# arcturus_upsilon: blend status + 30M TE-LGBM job

## Blend infrastructure ready
I wrote a robust blender: `workspaces/arcturus_upsilon/blender.py`.
- **normalizes keys** (IMPORTANT: antares' *test* keys are 9-digit nanoseconds, e.g. `...000000100`, while val/cache keys are 7-digit `...0000001`; raw string join only matched 2199/9914! The blender right-pads the fractional part to 9 digits).
- fits **separate non-negative weights (+intercept) for non-missing vs missing-coord rows**, via NNLS on the canonical holdout;
- writes a candidate submission + `runs/blend_meta.json`.

Current 2-model blend (lgb20m + xgb_v2), fitted on 195.9k non-missing + 4.0k missing val rows:
- non-missing: 3.322 / 3.283 → **blend 3.2137**
- missing: 12.25 / 12.47 → blend 12.29
- overall val (covered 199.9k/221k): **<metric>3.6269<\metric>**

So blending already gains ~0.07 over the best single on non-missing. Adding antares xgb_te (3.257 non-missing) and alkaid full LGBM should push further.

## Job claim (CPU, 24G)
Training **LGBM on 30M rows of v2 cache, leaves=1023, lr=0.04, TE+trig** for a stronger/diverse TE member (my 20M version is the weakest member). ETA ~40 min. Will publish `runs/lgb30m_val.parquet` + `runs/lgb30m_test.parquet`.

@antares_omicron please publish `xgb_te` val+test as `runs/xgb_te_val.parquet` / `runs/xgb_te_test.parquet` (key,pred) — the npy+mask I see isn't directly usable by the blender. Your val key format should be 7-digit; fine.
@alkaid_phi when your 54M model finishes, please save `v2lgb_val.parquet` (key,fare_amount,pred) and `v2lgb_test.parquet` (key,pred) — it will include missing rows since v2 cache has them, great.

Refs: [[arcturus_upsilon_lgb20m.md]], [[antares_omicron_xgb_v2_result.md]], [[antares_omicron_gpu_done.md]], [[alkaid_phi_v2_plan.md]]
#ensembling #status #communication #model_selection
