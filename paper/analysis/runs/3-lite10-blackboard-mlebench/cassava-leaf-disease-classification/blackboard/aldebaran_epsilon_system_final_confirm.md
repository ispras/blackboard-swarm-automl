# aldebaran_epsilon — SYSTEM final-phase confirmation ✅

Per [[2026-10-31_07-23-14_SYSTEM_final_phase.md]]: **no new experiments/trainings.** Final assembled & frozen.

- **Submission:** `final_solution/submission.csv`
- **md5:** `094dedcddb0117d0981b441aa56fceb3` — 2676 rows + header, `image_id,label`, IDs in `data/sample_submission.csv` order, labels ∈{0..4}, 0 nulls.
- **Model:** uniform 8-model hedged ensemble (`0.5·honest + 0.5·full-data-FT`), per-model best TTA (Swin late multi-scale .88999).
- **Shared-holdout (fold0) accuracy = <metric>0.89960</metric>** — independently recomputed, no test labels used.
- **Verification:** vega recompute = 0/2676 label mismatches; my recompute = 0/2676; format checks pass ([[vega_eta_FINAL_CONFIRMED.md]] [[vega_eta_final_reverify_swinms.md]]).
- **Bundle:** `submission.csv`, `README.md` (sources/revisions/licenses/purpose), `ensemble_meta.json`, `ens_{val,test}_probs.npy`, `code/` (21 scripts). Fallback prior freeze at `submission_pre_swinms.csv`.

**DONE — this is the agreed final solution.** I will only monitor integrity until hard_end.

#submission #status #communication #ensembling
