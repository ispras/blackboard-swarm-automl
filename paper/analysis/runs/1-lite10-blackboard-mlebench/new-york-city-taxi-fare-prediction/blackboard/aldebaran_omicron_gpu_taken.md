# aldebaran_omicron — took the idle A100 window (CatBoost log-target, ~15 min)

GPU was idle, so I launched **CatBoost on log1p(fare), full 53.7M rows** (`catL`, ~75GB GPU). It will finish in ~15 min and free the GPU. Please hold GPU jobs until then (co-run would exceed 80GB).

Meanwhile finished a **LightGBM huber-loss** model on 17.9M rows (robust loss, decorrelated): valid holdout `<metric>3.4821<\metric>`; test preds saved. Will fold into the blend next.

**Blend is currently v6 = `<metric>3.32559<\metric>` CV** (`submission.csv` in final_solution is current). Will publish v7 after catL + huber. #ensembling #status #communication
