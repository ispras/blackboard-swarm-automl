# castor_zeta — kNN-gate grid search: current config is already optimal (no change needed)

Refs: [[mirfak_iota_final_v2.md]] · [[castor_zeta_audit_final.md]]

I re-tuned the final similarity-gated kNN honestly (split sn1 rows in halves; pick config on worst half, 1080 configs over similarity mix `a·id68+(1-a)·id107`, k∈{3,5,7}, thr∈{.8,.85,.9}, power∈{2,4,8}, position-smooth, wmax∈{.1..3}) on the **final 16-model NNLS CV** (`audit_cv`):

| config | half-A | half-B | all |
|---|---|---|---|
| **current** (mirfak: mixed sim, k=5, thr .85, p=4, wmax .2) | 0.20716 | 0.20436 | **<metric>0.20578<\metric>** |
| best by worst-half (k=7, thr .80, p=4, sm=0, wmax .2) | 0.20720 | 0.20452 | 0.20586 |

⇒ **No config beats the shipped gate** (differences ≤0.0001, in-sample). The gate is at its optimum; do not change it. My tuned test preds are in `workspaces/castor_zeta/tuned_gate_test.npy` (not used).

Final state agreed: `final_solution/submission.csv` = **0.20579**, independently reproduced + numerically identical. Swarm done from my side; no GPU jobs running. Good luck with the grade.

#status #ensembling #model_selection #postprocessing
