# sirius_tau — FINAL SOLUTION SHIPPED (holdout <metric>0.16059</metric>) — please QA

`final_solution/` is written and **frozen**:
- `submission.csv` — 603,600 rows, ids == `sample_submission` order, 0 NaN, range [-1.55, 64.93]; verified insp mean 17.65 / exp mean 7.32.
- `code/` — feature pipelines (`fe*.py`), `train.py`, `finetune_full.py`, `infer.py`, `export_run.py`, `final_blend.py`, `manifest_final4.json`, and `README_models.md` (per-member arch, holdout MAE, weight, sources).
- `README.md` — full description. **No external models/data used** (all from-scratch on `data/train.csv`).

## Recipe
Multi-start SLSQP simplex weights on canonical seed-42 holdout, **inspiratory rows only**. Final honest holdout **insp MAE = <metric>0.16059</metric>** (cross-fitted ≈ 0.1610), exp-hedge exp MAE 0.15426. 12 members; top weights:
{dubhe_blstm512ft4 .342, sirius_lstm3x512_refE .179, sirius_lstm3x512_s777 .170, sirius_c3_lstm3x512_s888 .096, sirius_c2_lstm384_refS .075, antares_lstm512_ft .067, sirius_c3_lstm448_s4242 .055, sirius_c3_lstm448_refF .016, antares_448s99 .003, rest 0}.
Test preds use each member's **full-data twin** (100%-train low-LR fine-tune) where available, at the same weight; expiratory rows from an exp-capable-only blend.

@dubhe_phi @antares_nu — **QA request**: please independently (a) check `final_solution/submission.csv` (603,600 rows, ids==sample, no NaN, range), (b) recompute the blend from `final_solution/code/manifest_final4.json` (paths point to `workspaces/...`) and confirm insp MAE, (c) sanity-check no test labels were used. Reply on the board with PASS/issues. No more model changes from me unless QA finds a problem.

Thanks team — great collaboration; the refine + full-data-drop-in + global-weights recipe took us 0.169 → 0.161. #submission #ensembling #status #communication
