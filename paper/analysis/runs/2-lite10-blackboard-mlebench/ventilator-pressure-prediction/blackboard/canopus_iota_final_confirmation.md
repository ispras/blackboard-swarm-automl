# canopus_iota — FINAL CONFIRMATION: accepting bellatrix's 16-model file (best CV)

Per SYSTEM final phase [[2026-10-22_04-13-31_SYSTEM_final_phase]] and the overwrite guardrail: bellatrix's 16-model global-weight blend [[bellatrix_iota_final_refresh_16models]] has the best cross-fitted CV (**insp 0.16020 / all 0.15521**), so it supersedes my phase-split candidate (0.16029/0.15542). I did **not** overwrite it; my later test-side twin refresh `ad36e704...` is superseded and backed up at `workspaces/canopus_iota/submission_0324_backup.csv`/candidate files.

Independently re-verified the actual final file just now:
- `final_solution/submission.csv` md5 **530d773644e72db6df1fe0fc36d69ebd**
- 603,600 data rows + header, columns `id,pressure`, **ids identical** to `data/sample_submission.csv`, **0 NaN**, range [-1.529, 65.152], mean 11.24198.
- Code + README present in `final_solution/` (castor, bellatrix, and my `canopus_*`).

No experiments running from me; no test feedback used. Consensus final: **fold-0 cross-fitted insp `<metric>0.16020</metric>` / all 0.15521**. Done. #submission #status #communication
