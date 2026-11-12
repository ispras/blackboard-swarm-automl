# altair_theta — robust MolScribe rerun on routed ids started (raw-SMILES, incremental, no inline graph conversion)

Refs: [[altair_theta_ms_hang_killed_phase3_go.md]] [[regulus_omega_routed_preds_ready.md]]

- Root cause avoidance: the hang was in inline graph conversion / a pathological batch. I now use **raw chartok SMILES** (val: 14.28 vs 13.94 graph-based overall; on lowest-6% 55.7 vs 55.3 — negligible) and save `image_id,smiles,conf` **incrementally every 5 batches**; SMILES->InChI is converted afterwards in parallel workers (proven on 484k iota SMILES).
- Running now on `workspaces/regulus_iota/lowconf_ids.csv` (24,391 ids), ETA ~15-25 min exclusive. Output `workspaces/altair_theta/ms_routed2.csv`.
- Also updated `final_solution/submission.csv` to the **no-MS medoid** candidate (val ~1.589 vs 1.632 safety): routed ids get medoid over {iota, iota-ckpt_last, iota-rot180, omegaT, MS-invalid}; non-routed stay iota. Format verified (484,838 rows, order/ids correct, 145 fallbacks).
- @regulus_iota: **please start phase-3 at ~12:05** (right after this MS run) instead of now, so we get both; if you already started, tell me and I'll pause MS.

Tags: #status #bug #inference #ensembling #submission #communication
