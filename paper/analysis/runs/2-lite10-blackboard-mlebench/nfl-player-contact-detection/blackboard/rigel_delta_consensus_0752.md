# rigel_delta: 4-model consensus blend — JOINT OOF MCC 0.7521

Read [[castor_alpha_smoothing_finding.md]], [[achernar_sigma_pair_ground_v1_results.md]], [[rigel_delta_v2_postprocess_result.md]].

## Blend (shared castor folds, contact_id-keyed)
- Pair: my tracking-temporal (0.35) + achernar tracking+helmet v1 (0.65) -> pair MCC .7603 (blend weight selected on OOF)
- Ground: my tracking (0.15) + achernar tab (0.25) + **castor ground video (0.60)** [+ castor tab weight 0]
  - ground MCC **<metric>0.7033<\metric>** (single models: .459 / .966 AUC achernar / **.971 AUC castor video**)
- Centered smoothing: pair w=3, ground w=9, then jointly tuned thresholds.
- **JOINT OOF MCC <metric>0.7521<\metric>** (thr_pair .3627, thr_ground .5365).

Artifacts: `workspaces/rigel_delta/submission_consensus.csv` + `blend_consensus.json`; code `final_blend.py`.

## Calibration caveat (important)
Raw OOF thresholds applied to **fold-averaged test probs** under-predict (test pos rate 1.02% vs OOF optimum 1.23%/4.35% by subtype). I added a **rate-matched** submission variant (`submission_consensus_rate.csv`) that sets the test threshold per subtype to reproduce the OOF optimal positive rate. I lean toward the rate-matched version for the hidden test. Comments welcome.

## Waiting on / next
- my tracking+helmet pair model (5-fold, fold AUC up to .9983; ~10 min left) -> `oof_pair_h.parquet`
- achernar v2 pair/ground, castor pair-video full OOF/test
Then rerun `final_blend.py` with all models (weight search + smoothing + joint thresholds) and place best in `final_solution/`.

#ensembling #postprocessing #submission #status #model_selection
