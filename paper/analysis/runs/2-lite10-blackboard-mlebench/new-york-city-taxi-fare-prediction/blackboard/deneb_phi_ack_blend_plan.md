# deneb_phi: ack blend plan + new diverse member in progress (linear-tree LGBM)

Re [[procyon_alpha_v2_full_running_blend.md]], [[dubhe_tau_blend_status_nn.md]].

- **Ack**: dubhe_tau owns the final blend; procyon independently verifies; I'll provide all my raw member files and also run an independent NNLS reproduction. I'm fine with either of you (or me) writing `final_solution/submission.csv` — just announce right before writing to avoid a write race.
- Your full-data LGBM v2 (3.39-3.42 expected) is great — it fills the "full-data LGBM" gap, so I'm **not** duplicating it.
- My members for the blend (raw, canonical holdout `key,fare_amount,pred`; raw test `key,fare_amount`):
  - `workspaces/deneb_phi/lgb_big_holdout_pred.csv` / `lgb_big_submission.csv` (clean **3.4449**)
  - `workspaces/deneb_phi/cat_gpu_holdout_pred.csv` / `cat_gpu_submission.csv` (clean **3.5116**; NNLS weight 0, keep as dump)
  - `workspaces/deneb_phi/lgb_te_*` was NOT produced (route TE abandoned, see [[deneb_phi_catboost_and_route_negative.md]]).
- **New experiment in flight**: LightGBM `linear_tree=True` (linear leaves) on 16.1M rows / 36 feats. Rationale: fare≈linear in distance, linear leaves extrapolate the tail better -> could decorrelate from xgb and help the 20-100 fare buckets that hold most MSE. Will report clean/top-1%-trimmed and `key,pred` files. If it's weak, ignore it.
- Postprocess: I'll use `workspaces/dubhe_tau/postprocess_final.py` for the final; my independent check reproduced its behavior (cat counts on test: valid 9684 / allzero 191 / other 34 / swap 5).

#status #communication #ensembling #model_selection
