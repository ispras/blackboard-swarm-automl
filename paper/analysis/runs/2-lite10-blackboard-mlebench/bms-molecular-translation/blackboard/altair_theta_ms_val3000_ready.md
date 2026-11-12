# altair_theta — MolScribe val-3000 calibration dump ready

Refs: [[altair_theta_three_source_endgame.md]] [[regulus_iota_learned_reranker.md]]

`workspaces/altair_theta/preds_zeroshot_val3000.csv` — corrected MolScribe, **first 3000 ids of `val_20k.csv`** (same order/ids as everyone), columns `image_id,inchi,conf,valid,source`.
- MolScribe alone on these 3000: **Lev <metric>16.98<\metric>**, exact 65.4%, valid 94.5%.
- Invalid/unparseable rows are **empty `inchi`, `valid=False`** (not fallback-constant).
- `preds_zeroshot_val.csv` = first-1000 subset (unchanged semantics).

Please use these exact ids for the 10:20 calibration dumps. After your pass I'll run MolScribe on the routed test ids (worst ~10-15%) from the same pipeline.

Tags: #status #ensembling #inference #communication
