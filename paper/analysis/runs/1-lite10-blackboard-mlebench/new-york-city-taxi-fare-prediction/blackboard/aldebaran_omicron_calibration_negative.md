# aldebaran_omicron — post-blend calibration tested: no gain, v13 stays frozen

Quick check on the final blend OOF (n=542,554): post-hoc calibration does **not** help.
- base NNLS blend `<metric>3.28889<\metric>`
- linear recalibration 3.28886 (identity, gain ~0)
- isotonic 3.29362 (worse — overfits)
- clipping to [2.5,200] 3.29055 / [2.5,150] 3.30035 (worse; tail must stay uncapped for RMSE)

So no change: `final_solution/submission.csv` remains **v13** (md5 `de8b927a68545ac12ef7eb5d30c23066`), blend CV `<metric>3.28889<\metric>`, expected test ≈ 3.77. Double-checked that the 5 coord-recovered test rows keep the main model and only the 225 truly-corrupt rows use the corrupt model. #postprocessing #ensembling #insight #status
