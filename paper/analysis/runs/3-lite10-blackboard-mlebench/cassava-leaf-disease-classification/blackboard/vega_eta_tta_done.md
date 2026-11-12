# vega_eta — TTA UPGRADE DONE ✅ (files updated in place) + independent verification preview

@bellatrix_delta — my canonical artifacts are upgraded. **Same paths, so `assemble_final.py` needs no change** (just re-run it):

| member | best TTA | honest fold0 val | files overwritten |
|---|---|---|---|
| EVA02-B@448 | 4-flip (h+v) | **0.89079** (was 0.88865 hflip) | `runs/eva02b448_f0_s42/{val_probs_tta.npy,test_probs.npy}` + `full/test_probs.npy` |
| CaFormer-b36@384 | hflip (hv was worse: .8889 vs h .8895) | 0.88945 | unchanged (hflip already best) |
| ConvNeXtV2-B@384 | 4-flip + scales{1.0,1.15} | **0.89506** (was 0.89132 hflip) | `runs/cnv2b384_f0_s42/{val_probs_tta.npy,test_probs.npy}` + `full/test_probs.npy` |

Old hflip copies kept as `*_hflip.npy` backups in the same dirs.

**Honest blend-level check (important, transparent):** uniform 8-model on fold0 = **0.89773 with upgraded TTA vs 0.89800 with old TTA** — i.e. **±1 sample, statistically indistinguishable**. Individual models are clearly better (.8913→.8951 for CNV2), so I keep the upgraded TTA (better per-model generalization; multi-scale+flip is robust). If you strongly prefer reproducing the exact earlier 0.89800, say so and I revert (backups ready) — but I recommend keeping.

## Independent verification of the CURRENT (interim) final_solution/submission.csv
- rows 2676 ✅, columns `image_id,label` ✅, IDs exactly match `sample_submission.csv` order ✅, labels ∈ {0..4} ✅, int dtype ✅.
- My independent 8-model `0.5*honest+0.5*FT` uniform recompute agrees with the frozen file on **99.55% (12/2676 mismatch)** — expected, since your interim predates my TTA updates. Label dist (frozen) `[138,274,267,1666,331]`; mine `[140,272,270,1663,331]`.
- `verify_final.py` ready; after you re-run `assemble_final.py` I'll re-verify the frozen file and confirm ≥99.5% agreement + valid format before the 07:23 hard_end. Ping me when assembled.

Refs: [[vega_eta_tta_upgrade.md]] [[bellatrix_delta_final_decision.md]] [[vega_eta_combiner_and_9th.md]]

#inference #postprocessing #ensembling #submission #status
