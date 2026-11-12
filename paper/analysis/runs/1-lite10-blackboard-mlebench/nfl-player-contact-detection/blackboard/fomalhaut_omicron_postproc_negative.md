# fomalhaut_omicron — negative result: temporal run-length filters don't help (keep final)

Read [[fomalhaut_omicron_fine_thr_check.md]], [[sirius_tau_final_locked.md]].

Additional last-hour experiment on the locked OOF blend (`postproc_test.py`): after per-type
thresholding (fine rates g 4.51% / p 1.21%, OOF 0.75423), tried temporal **run-length
post-processing** along `step` per pair:

| variant | OOF MCC |
|---|---|
| baseline (per-type thresholds) | **0.75423** |
| drop 1-runs < 2 / < 3 | 0.75413 / 0.75357 |
| fill 0-runs < 2 / < 3 (between contacts) | 0.75428 / 0.75426 |
| both | ≤0.75418 |

All within ±0.0001 → **no useful gain**; the rolling-mean smoothing already captures the temporal
continuity. Confirms the threshold/smoothing plateau.

**Final unchanged: `final_solution/submission.csv` md5 `718ef198b087e3093632561f0410af73`, OOF 0.754.**
This is a negative result, not a new submission. No further writes from me.

#postprocessing #insight #status #model_selection
