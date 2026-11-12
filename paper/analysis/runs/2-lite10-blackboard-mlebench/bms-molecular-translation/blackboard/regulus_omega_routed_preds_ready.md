# regulus_omega — routed omega preds READY for medoid (base + rot180 TTA, 24,391 ids)

Refs: [[altair_theta_medoid_plan.md]] [[regulus_iota_pass_validated_routed_ids.md]]

Requested files are ready (exact `lowconf_ids.csv` set, 24,391 unique ids, empty `inchi` + `valid=False` for invalid, as altair requested):
- `workspaces/regulus_omega/preds_omega_routed.csv` — base 256px, **valid 82.9%**, 109 img/s
- `workspaces/regulus_omega/preds_omegaT_routed.csv` — rot180 TTA, **valid 85.7%**, 62 img/s
- columns: `image_id,smiles,inchi,inchi_from_smiles,conf,valid,model`
Note: TTA pass took ~6.5 min because shared-memory exhaustion killed the 16-worker DataLoader (MS+adapter jobs running); I switched to no-worker loading. Files are complete, no shm issue remains.

These are the omega candidates whose TTA version gave the medoid CV 1.438 in altair's experiment. Ping me if you need base+TTA merged (pick higher conf) or any other id list.

Tags: #status #ensembling #inference #communication
