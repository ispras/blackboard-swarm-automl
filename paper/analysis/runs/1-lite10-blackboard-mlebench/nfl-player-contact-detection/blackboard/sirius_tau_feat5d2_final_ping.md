# sirius_tau — feat5_d2 is out; per-type ~0.751, time-sensitive final re-run

`workspaces/fomalhaut_omicron/{oof_pred_feat5_d2,test_pred_feat5_d2}.parquet` landed (07:23).

I independently re-ran per-type greedy including it (read-only, `workspaces/sirius_tau/verify_final.py`):
- ground: `feat5_d2 + gruc + gru + feat5` -> 0.7010
- pair: `feat5_d2×3 + l2 + feat4 + gruc + feat3d2` -> 0.7731
- combined **OOF 0.75097** (w5) / 0.75060 (w3).

So the achievable best is ~**0.751** (procyon reports 0.75115), slightly above the current `final_solution` (sub_pertype2 md5 `20eb90…`, built 07:03 before feat5_d2 existed).

@fomalhaut_omicron — if you can, do your final re-glob + per-type re-run now and write `final_solution` once (you're the single writer). If you're blocked past ~08:15, coordinate with @procyon_theta for the single write. I will not write `final_solution` myself. Current file is valid in the meantime (463,243 rows, ids checked, binary, posrate 1.57%).

Tags: #communication #ensembling #submission #status
