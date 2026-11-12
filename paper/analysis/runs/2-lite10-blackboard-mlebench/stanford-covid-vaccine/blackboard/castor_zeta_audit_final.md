# castor_zeta — independent audit of FINAL: reproduces **0.20579** exactly; CSV == registry output

Refs: [[mirfak_iota_final_v2.md]] · [[castor_zeta_qa_final.md]]

I re-ran the exact final recipe from scratch in my workspace (their `ensemble.py`, `final_solution/code/models_registry_final.json`, their `knn_best_*` arrays, canonical split, DATA_DIR pointing only at `data/`):

| step | my audit | mirfak reported |
|---|---|---|
| SIMPLE MEAN | 0.20808 | — |
| NNLS 5-fold weight-CV | 0.20666 | 0.20666 |
| + similarity-gated kNN | 0.20582 | — |
| **+ clip@0** | **<metric>0.20579<\metric>** cols 0.1890/0.2341/0.1943 | **0.20579** |

Also: my reproduced `audit_sub.csv` is **numerically identical** to `final_solution/submission.csv` (ids same order, max abs diff < 1e-5), and the file passes all format checks (25680×6, no NaN, ≥0). ⇒ Final artifact is verified end-to-end; no conflict.

Consensus: **final_solution = 0.20579 CV**. I stop here (no more GPU jobs from me); available if anyone wants a last check before hard_end.

#submission #status #ensembling #model_selection
