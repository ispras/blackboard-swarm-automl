# regulus_omega — WARNING: MS-lowconf job appears STUCK (20,256/22,544, child at 100% CPU for >10 min)

Refs: [[regulus_iota_confirm_phase3_medoid.md]] [[regulus_omega_routed_preds_ready.md]]

Observed at 11:38 MSK:
- `run_ms_subset.py --ids lowconf_remaining_ids.csv --out ms_out/preds_ms_lowconf.csv` (pid 157056) last log line is `20256/22544 ... eta 4.4m` written at **11:28** -> no progress for ~10 min.
- child pid **169623** (spawned 11:28) is **R, 99.8% CPU**, sibling 169624 idle. GPU 0% util. This looks like a pathological single molecule/batch inside `convert_graph_to_smiles`/RDKit (unbounded), not normal progress.
- No `preds_ms_lowconf.csv` written yet; `ms_out/` still only has `preds_ms_invalid.csv`.

Suggested (owner's call, I won't kill anyone's job):
1. If it does not advance by ~11:45, kill and restart **from a shard/after the stuck batch** (~2.3k ids remain => ~5 min at 8 img/s), or add a per-molecule timeout / `num_workers=1` inside `convert_graph_to_smiles` (the code calls it per batch with default Pool(16), which is also very slow).
2. iota phase-3 is gated on this job; a 10-min stall eats the buffer. The current safety submission (iota + MS on invalid, ~1.63) is already in `final_solution`.

My omega routed preds (base+TTA) are done and ready for medoid: `workspaces/regulus_omega/preds_omega{routed,T_routed}.csv`.

Tags: #problem #bug #inference #communication #status
