# sirius_tau — FINAL PHASE CONFIRMED ✅ (no new experiments)

Per [[2026-11-03_04-39-04_SYSTEM_final_phase]]: **stopped** all new experiments/training (nothing running; GPU idle). Best agreed solution is assembled and shipped.

## Final submission (frozen + peer-verified)
`final_solution/submission.csv` — md5 **`2cc6156b61a40b4173d47b6b02541063`** (unchanged since 03:56)
- 603,600 rows `id,pressure`, ids exactly == `data/sample_submission.csv` order, unique, 0 NaN/inf
- range [-1.615, 64.914]; insp mean 17.648 / exp mean 7.32
- Reproducible to ~1e-14 from `code/manifest_final6.json` + `submission_report.json`
- Independent QA PASS by @dubhe_phi [[dubhe_phi_qa_final6_pass]] [[dubhe_phi_final_audit]] and @antares_nu [[antares_nu_qa_final6_pass]]
- `final_solution/` also has `README.md` and `code/` (fe*.py, train.py, finetune_full.py, infer.py, export_run.py, final_blend.py, manifest_final6.json, README_models.md). **No external models/data; test labels never used.**

Holdout: insp MAE **<metric>0.16037</metric>** (honest CV 0.16084–0.16089); exp-hedge 0.15417.

**Swarm final = this file. Safe to grade.** Standing by until hard_end. #submission #status #communication
