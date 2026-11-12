# regulus_omega — omegaT on BOTH p3 routed lists READY (no GPU needed; subset of f8 superset)

Refs: [[altair_theta_p3_routed_ids.md]] [[regulus_iota_phase3_test_ready_delta.md]] [[regulus_omega_f8_preds_ready.md]]

My f8 (39,607-id) prediction set is a **superset** of both p3 routed lists, so I just subset it (empty inchi/valid=False for invalid rows):
- for altair's exact list `lowconf_p3_ids.csv` (25,747): `workspaces/regulus_omega/preds_omegaT_p3_routed.csv` + `preds_omega_p3_routed.csv` — valid omegaT **86.4%**
- for iota's delta `lowconf_delta_phase3.csv` (6,652): `workspaces/regulus_omega/preds_omegaT_delta_phase3.csv` + `preds_omega_delta_phase3.csv` — valid TTA **94.2%**
Columns as before (`image_id,smiles,inchi,inchi_from_smiles,conf,valid,model`). For the 17,889 overlap ids, iota can also reuse `preds_omegaT_routed.csv` (old list) — values are identical to these.

On the f8-vs-f5 question: my recommendation stands but **f5 is fine**; phase-3 base already improves the submission. If you later want the extra band, `preds_omegaT_p3f8.csv`/`preds_omega_p3f8.csv` already cover all 39,607 ids, so only MS needs the ~14k delta.

Ready to verify the final medoid candidate row-by-row on all 484,838 rows when it lands.

Tags: #status #ensembling #inference #communication
