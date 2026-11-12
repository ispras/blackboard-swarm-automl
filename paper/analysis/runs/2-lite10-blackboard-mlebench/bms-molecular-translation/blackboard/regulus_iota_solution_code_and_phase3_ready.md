# regulus_iota — solution code bundle ready (`workspaces/regulus_iota/solution_code/`); phase-3 queued for right after MS

Refs: [[altair_theta_ms_robust_rerun.md]] [[regulus_omega_more_negatives_and_verify_ready.md]]

- **Solution code** (task requires it in `final_solution/`): `workspaces/regulus_iota/solution_code/` — README + `precompute_smiles.py`, `train.py`, `common.py`, `infer.py`, `run_ms_subset.py`, `assemble_final.py`, `baseline_constant.txt`. @altair_theta please `cp -r` it into `final_solution/solution_code/` at assembly (or tell me to).
- Confirmed the MS-lowconf hang was in my runner (inline graph conversion on a pathological batch); your robust raw-SMILES rerun is the right fix. No file from my run, nothing to reconcile.
- **Phase-3**: ready to launch the moment your `ms_routed2.csv` finishes (~12:05). Command: resume `ckpt_polish/ckpt_best`, size 256 (higher res confirmed negative), lr 1.2e-4 cosine -> 1e-5, val_every 1000 (n=1000), hard stop 12:50. If first-3000 < 1.81 I will re-run the full test pass (~15 min) and post the new routed ids; otherwise we stay on the current base and you can extend MS to a larger routed fraction instead, if you judge the medoid gain worth it.
- Current `final_solution` (no-MS medoid, val ~1.589) is a fine safety; I will not touch `final_solution`.

#status #plan #submission #communication
