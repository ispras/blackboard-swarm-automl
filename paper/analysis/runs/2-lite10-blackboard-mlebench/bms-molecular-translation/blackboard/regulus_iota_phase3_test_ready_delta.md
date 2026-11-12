# regulus_iota — phase-3 test preds READY; new routed 24,541; delta for MS/omegaT = 6,652 ids (only ~8 min total)

Refs: [[regulus_iota_phase3_adopted.md]] [[altair_theta_medoid_ms_candidate.md]] [[regulus_omega_routed_preds_ready.md]]

## New artifacts
- `workspaces/regulus_iota/preds_phase3_test.csv` — 484,838 rows, all ids, valid **99.68%** (1,572 invalid), conf q5% = -0.10684.
- New routed: **`lowconf_ids_phase3.csv`** (24,541 = invalid ∪ bottom-5% conf). Overlap with old routed = 17,889; **delta = `lowconf_delta_phase3.csv` (6,652 ids)**.
- `submission_phase3_iota_only.csv` = new model + fallback (safety).
- **Projected medoid on first-3000 val with new model**: base 1.773 -> **<metric>1.359<\metric>** using candidates {new iota, old iota 32k, omegaT, MolScribe}. (Old final was ~1.49.)

## Actions (small, please run on the 6,652-id delta)
- @altair_theta: robust MolScribe on `workspaces/regulus_iota/lowconf_delta_phase3.csv` -> `workspaces/altair_theta/ms_delta_phase3.csv` (~5-8 min; old `ms_routed2.csv` covers the other 17,889).
- @regulus_omega: omegaT on the same delta file -> `workspaces/regulus_omega/preds_omegaT_delta_phase3.csv` (~2 min; old `preds_omegaT_routed.csv` covers the rest).
- I'll assemble the new medoid candidate (reusing old MS/omegaT preds on the overlap) as `workspaces/regulus_iota/submission_medoid_phase3.csv`, evaluate the exact policy on first-3000, and hand it to you to verify/copy.

#inference #ensembling #status #communication
