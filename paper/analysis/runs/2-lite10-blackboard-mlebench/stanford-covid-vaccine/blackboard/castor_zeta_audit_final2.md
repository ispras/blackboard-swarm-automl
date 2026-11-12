# castor_zeta — audit of FINAL v3 (15:44): reproduces **0.20576** exactly ✅

Refs: [[mirfak_iota_final_v2.md]] · [[castor_zeta_audit_final.md]] · [[castor_zeta_gate_verified.md]]

Re-ran the updated compact registry (`final_solution/code/models_registry_final.json`, 16 entries incl. 11-seed struct_all5) with their exact `ensemble.py` + `knn_best_*` gate in my workspace:
- NNLS CV 0.20664 → **gated+clip <metric>0.20576<\metric>** cols 0.1890/0.2340/0.1942 — matches their README 0.20576 exactly.
- My reproduced submission is numerically identical to `final_solution/submission.csv` (same ids order, max abs diff < 1e-9).
- Format re-checked: 25680×6, no NaN, all ≥0, 240 ids × 107 rows.
Minor note: registry lists `my_struct_{scored,all5,huber}` twice (build_final appended duplicates). Harmless (identical arrays) but can be de-duplicated for cleanliness — no need to re-merge.

Consensus: final = **0.20576** CV, independently verified. Done from my side.

#submission #status #ensembling
