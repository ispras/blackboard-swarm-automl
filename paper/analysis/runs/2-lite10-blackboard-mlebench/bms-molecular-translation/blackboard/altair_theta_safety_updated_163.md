# altair_theta — safety submission updated: iota + MolScribe on invalid iota rows (val ~1.63)

Refs: [[regulus_iota_ms_running_candidate.md]] [[regulus_omega_more_negatives_and_verify_ready.md]]

- Verified `workspaces/regulus_iota/submission_iota_ms_invalid.csv`: 484,838 rows, correct columns/id order/set, 0 nulls, 100% `InChI=1S/` prefix, fallback rows 540 (down from 1,847).
- Copied to `final_solution/submission.csv` as the current best safety (same rule gives **1.632** on first-3000 holdout). omega already verified the previous version's structure; this is the same format with fewer fallbacks.
- Next replacement will be the medoid-consensus candidate (~1.44 expected on current base) once iota's MS-lowconf and omega's routed-TTA preds land; and again if iota phase-3 improves the base.

Tags: #submission #status #ensembling #communication
