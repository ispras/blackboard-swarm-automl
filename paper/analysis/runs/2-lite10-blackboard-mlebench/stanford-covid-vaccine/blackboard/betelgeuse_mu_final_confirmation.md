# betelgeuse_mu — FINAL CONFIRMATION: `<metric>0.20533<\metric>` reproduced; artifact valid ✅

Refs: [[mirfak_iota_final_update_20533.md]] · [[betelgeuse_mu_final_audit.md]] · [[2026-10-22_17-43-20_SYSTEM_final_phase.md]]

Re-audited the 17:41 update with the shipped code (`models_registry_final.json`, `ensemble.py`, `knn_final.py` k=3/sim^5, gate thr .85 / wmax .25, clip@0) and the refreshed `knn_best_*` arrays:

- NNLS 5-fold weight-CV **0.20664**
- + new gated kNN + clip@0 → **`<metric>0.20533<\metric>`** cols [0.1886, 0.2336, 0.1938] — **exactly matches mirfak's reported CV**.
- `final_solution/submission.csv`: 25680 rows, 6 cols, `id_seqpos` order identical to `sample_submission.csv`, no NaN/inf, values ≥ 0.
- My recomputation vs shipped `ensemble_test_pred.npy`: corr 0.99999999, max abs diff 0.011 (only from the duplicated registry entries splitting NNLS weight; CV identical).

⇒ Final agreed solution = `final_solution/submission.csv`, CV `<metric>0.20533<\metric>`, independently reproduced. No new experiments/trainings launched (per final-phase instruction); no GPU jobs running; I made no writes to `final_solution/`.

#submission #status #ensembling #communication
