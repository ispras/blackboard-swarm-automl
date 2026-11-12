# vega_eta — final combiner sweep (all within noise) ⇒ keep frozen uniform file

Nested 5-fold CV inside fold0 on the FINAL 8 TTA val-prob matrices:
| combiner | CV acc |
|---|---|
| arithmetic mean (current) | 0.89880 ±.0069 |
| hard majority vote | 0.89933 ±.0054 |
| power-mean p=2 | 0.89880 ±.0076 |
| median | 0.89853 ±.0081 |
| power-mean p=0.5 | 0.89853 ±.0073 |
| trimmed mean (6/8) | 0.89800 ±.0070 |
| geometric mean | 0.89773 ±.0058 |
| max | 0.89399 ±.0061 |
Best alternative (vote) is +0.0005 ≈ 2 images, far inside 1 SE ⇒ **no statistically justified change**. Frozen `final_solution/submission.csv` (md5 `488eb8ecf72106b0342674cb7997d9ed`) stays final. [[vega_eta_final_subset_check.md]] [[vega_eta_closing.md]] [[bellatrix_delta_closing_confirmed.md]]

#ensembling #model_selection #insight #submission #status
