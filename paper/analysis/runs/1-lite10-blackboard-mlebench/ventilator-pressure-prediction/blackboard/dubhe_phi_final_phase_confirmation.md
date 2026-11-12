# dubhe_phi — FINAL PHASE CONFIRMED ✅ (nothing new launched)

Per SYSTEM final-phase instructions [[2026-11-03_04-39-04_SYSTEM_final_phase]]: no new experiments/models. Best agreed solution is assembled and shipped.

## Final submission (frozen, verified)
`final_solution/submission.csv`
- md5 **`2cc6156b61a40b4173d47b6b02541063`** (unchanged since 03:56)
- 603,600 rows, `id,pressure`, ids exactly == `data/sample_submission.csv` order, unique
- 0 NaN / 0 inf, range [-1.615, 64.914], insp mean 17.648 / exp mean 7.32
- Reproducible from `code/manifest_final6.json` + `submission_report.json` to ~1e-14 (independent QA by dubhe + antares; peer reviews [[dubhe_phi_qa_final6_pass]] [[antares_nu_qa_final6_pass]] [[sirius_tau_consensus_final]])
- `final_solution/` also contains `README.md`, `code/` (fe*.py, train.py, finetune_full.py, infer.py, export_run.py, final_blend.py, manifest_final6.json, README_models.md). **No external models/data; no test labels.**

## Consensus on performance/saturation
- Honest seed-42 holdout insp MAE **0.16037** (cross-fitted 0.16084–0.16089); exp-hedge exp MAE 0.15417.
- Extra members (27-member pool) and all postprocessing calibrations gave no gain ([[sirius_tau_subset_cv]] [[antares_nu_postproc_negative_result]] [[dubhe_phi_final_audit]]).

**Swarm final = this file. Safe to grade.** #submission #status #communication #ensembling
