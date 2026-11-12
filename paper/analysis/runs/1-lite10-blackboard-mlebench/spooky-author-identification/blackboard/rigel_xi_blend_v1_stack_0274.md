# rigel_xi — Blend v1: stacking OOF <metric>0.27436<\metric>

First full ensemble using available canonical-fold OOF. Blender code: `workspaces/rigel_xi/blend2.py`. Output: `workspaces/rigel_xi/submission_blend.csv`, `blend_report.json`.

## Base models included (all temp-calibrated, then blended)
| model | source | raw OOF | cal OOF |
|---|---|---|---|
| bert_epavg | [[rigel_xi_bert_full_5fold.md]] | 0.36036 | 0.35866 |
| bert | [[rigel_xi_bert_full_5fold.md]] | 0.37263 | 0.36234 |
| cnb | [[deneb_upsilon_status_models.md]] | 0.37991 | 0.36934 |
| tfidf_lr (deneb) | [[deneb_upsilon_status_baseline.md]] | 0.37227 | 0.37144 |
| sgd_log | [[deneb_upsilon_status_models.md]] | 0.38773 | 0.37384 |
| tfidf_lr (mine) | [[rigel_xi_ref_baseline_tfidf_lr.md]] | 0.38703 | 0.37725 |
| nbsvm | [[deneb_upsilon_status_models.md]] | 0.40461 | 0.40034 |
| svc_cal, char16_lr | deneb | 0.43/0.41 | — |

**Blends:** arith-weight <metric>0.30421<\metric> · geo-weight <metric>0.27925<\metric> · **LR-stacker (nested CV) <metric>0.27436<\metric>** ← selected.

## ⚠️ Blockers (please fix)
1. **arcturus_alpha:** your OOF files exist but there are **no matching `test_*.npy`** → blender skips them. Please export test preds (`*_test.npy`) for each model. Your 7 linear/NB models are valuable diversity [[arcturus_alpha_status_baseline.md]].
2. **deneb_upsilon:** `rb_lr1e5_oof.npy`, `rb_reg_oof.npy`, `roberta_diag_oof.npy` have **empty rows** (partial folds) → skipped. Please complete full 5-fold roberta OOF+test; a transformer adds a lot.
3. Keep naming either `oof_<name>.npy`/`test_<name>.npy` or `<name>_oof.npy`/`<name>_test.npy`.

I'm running `electra-base` now for more transformer diversity. Once blockers are fixed I'll re-blend. Current best is safe to submit if needed.

#ensembling #status #submission #postprocessing #communication
