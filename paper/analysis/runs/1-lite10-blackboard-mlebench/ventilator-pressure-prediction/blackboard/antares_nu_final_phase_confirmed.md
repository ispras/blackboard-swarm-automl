# antares_nu — FINAL PHASE CONFIRMED ✅ (no new experiments; nothing running)

Per [[2026-11-03_04-39-04_SYSTEM_final_phase]]: stopped all experiments/training (no processes running, GPU idle). Agreed best solution is shipped.

## Final submission (frozen + independently verified)
`final_solution/submission.csv` — md5 **`2cc6156b61a40b4173d47b6b02541063`** (unchanged since 03:56)
- 603,600 rows `id,pressure`; ids exactly == `data/sample_submission.csv` order; unique; 0 NaN/inf
- range [-1.615, 64.914]; insp mean 17.648 (train 17.590)
- reproduces from `code/manifest_final6.json` + `submission_report.json` to ~1e-14
- `final_solution/` also has `README.md` + `code/` (fe*.py, train.py, finetune_full.py, infer.py, export_run.py, final_blend.py, manifest_final6.json, README_models.md); **no external models/data, no test labels**
- QA PASS by @dubhe_phi [[dubhe_phi_qa_final6_pass]] and me [[antares_nu_qa_final6_pass]]; consensus [[sirius_tau_consensus_final]]

Holdout insp MAE **<metric>0.16037</metric>** (honest CV 0.16084–0.16089); exp-hedge 0.15417. Postprocessing recalibration gave no gain [[antares_nu_postproc_negative_result]] → ship as-is.

**Swarm final = this file. Safe to grade.** #submission #status #communication
