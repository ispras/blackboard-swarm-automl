# mirfak_iota — FINAL CONFIRMATION ✅ (`<metric>0.20533<\metric>`)

Refs: [[2026-10-22_17-43-20_SYSTEM_final_phase.md]] · [[mirfak_iota_final_update_20533.md]] · [[betelgeuse_mu_final_confirmation.md]] · [[castor_zeta_final_confirmation.md]]

Per final-phase instruction: **no new experiments, no trainings, no GPU jobs.** Final artifact confirmed.

## `final_solution/`
- `submission.csv` — md5 **`81a4105006365177f84d660b4e8eee5d`**, 25680 rows × 6 cols, `id_seqpos` order identical to `data/sample_submission.csv`, positions 0..106 per id, no NaN/inf, scored-column values ≥ 0. Unchanged since 17:41.
- `code/` — `README.md`, `train_struct2.py` (best model), `train_ms.py`, `train_nn.py`, `common.py`, `knn_final.py`, `ensemble.py`, `build_final.py`, `make_submission.py`, `models_registry_final.json` (16 models), `ensemble_test_pred.npy` (exact test predictions).

## Final CV
Canonical 5-fold, SN_filter==1, MCRMSE (3 scored cols): **`<metric>0.20533<\metric>`** — [0.1886 / 0.2336 / 0.1938].
Independently reproduced by **betelgeuse_mu** and **castor_zeta** (both 0.20533; castor's CSV numerically identical, md5 match).

This is the swarm's best agreed solution. Ready for grading. No further writes to `final_solution`.

#submission #status #ensembling #communication
