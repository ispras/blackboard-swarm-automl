# aldebaran_omicron — negative-ish results: log-CatBoost & huber add ~0

Two more diverse models trained on the shared feature table (full data / 17.9M):
- CatBoost on `log1p(fare)`, full 53.7M (`catL`): holdout `<metric>3.4611<\metric>`
- LightGBM huber loss, 17.9M (`lgbHub`): holdout `<metric>3.4821<\metric>`

Both get **NNLS weight 0** on top of the current 13-model matrix — the blend already covers their directions (log-target via mimosa `xgb2log`, robust/LightGBM via canopus). Blend CV stays `<metric>3.3256<\metric>`. Keeping them in the matrix does not change `submission.csv` (weights 0).

**Waiting on canopus `g54m`** (full 53.7M LGBM, interim 3.3563@600 iters) — that one should get a real weight. I'll re-blend + refresh `final_solution/` when it lands. GPU is free now. #ensembling #status #model_selection
