# vega_eta — FINAL CONFIRMATION (per SYSTEM final-phase instruction) ✅

Final phase acknowledged (hard_end 07:53). No new experiments/trainings — **final solution is assembled, verified, and frozen.**

- **Submission:** `final_solution/submission.csv`
- **md5:** `094dedcddb0117d0981b441aa56fceb3`
- **Contents:** 2676 data rows + header; columns `image_id,label`; IDs exactly in `data/sample_submission.csv` order; labels ∈ {0,1,2,3,4} int; 0 nulls.
- **Model:** uniform 8-model hedged ensemble (`0.5·honest + 0.5·full-data-FT`), per-model best TTA:
  EffNet-B5@512, EffNet-B5@640, ConvNeXt-B@384, EVA02-B@448, CaFormer-b36@384, Swin-B@384 (late multi-scale TTA .88999), ViT-B@384, ConvNeXtV2-B@384 (.89506).
- **Shared-holdout (fold0) accuracy = <metric>0.89960</metric>** (independent recompute; no test labels used for any fitting).
- **Independent verification:** my recompute from raw member `.npy` = **0/2676 label mismatches**; format checks pass. Also verified by aldebaran/bellatrix.
- **Bundle complete:** `submission.csv`, `README.md` (model IDs/HF repos/revision shas/licenses/purpose), `ensemble_meta.json`, `ens_{val,test}_probs.npy`, `code/` (all agents). Previous freeze preserved at `submission_pre_swinms.csv` (md5 `488eb8…`) as a one-`cp` fallback.

**Verdict: DONE. This is the agreed final solution.** No further changes from me; I will only monitor integrity until hard_end.

Refs: [[aldebaran_epsilon_late_final.md]] [[vega_eta_final_reverify_swinms.md]] [[bellatrix_delta_lastmile_negative.md]] [[2026-10-31_07-23-14_SYSTEM_final_phase.md]]

#submission #status #communication #ensembling
