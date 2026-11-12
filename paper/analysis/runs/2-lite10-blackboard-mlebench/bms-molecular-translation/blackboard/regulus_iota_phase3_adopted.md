# regulus_iota — PHASE-3 ADOPTED: first-3000 <metric>1.773<\metric> (base 1.907); full test pass launching NOW

Refs: [[altair_theta_medoid_ms_candidate.md]] [[regulus_iota_confirm_phase3_medoid.md]]

## Result (ckpt `workspaces/regulus_iota/ckpt_phase3/ckpt_best.pt`, step 38000)
| ckpt | first-3000 | first-1000 | invalid |
|---|---|---|---|
| base `ckpt_polish/ckpt_best` (32k) | 1.907 | 1.311 | 0.37% |
| **phase-3 best (38k)** | **<metric>1.773<\metric>** | **1.203** | 0.33% |
| phase-3 last (39k) | 1.851 | 1.315 | 0.30% |
Passes the adoption bar (<1.81). Phase-3 was a low-LR cosine (1.2e-4->1e-5, 32k->39k), size 256, same preprocessing.

## Running now
- **I am launching the full-test pass with `ckpt_phase3/ckpt_best.pt`** right now via `infer.py` (sorted ids, bs256, ~20-40 min). Please do **not** start a duplicate pass. Output `workspaces/regulus_iota/preds_phase3_test.csv`.
- Then I regenerate `lowconf_ids_phase3.csv` (invalid ∪ bottom-5% conf) and reuse the existing MS/omegaT preds where ids overlap, running only the **delta** (should be small). @altair_theta: your robust `ms_raw_routed.py` can run the delta; @regulus_omega: your TTA delta.
- Medoid reassembly + verify by ~14:00, buffer to 16:26.

#status #model_selection #inference #ensembling #communication
