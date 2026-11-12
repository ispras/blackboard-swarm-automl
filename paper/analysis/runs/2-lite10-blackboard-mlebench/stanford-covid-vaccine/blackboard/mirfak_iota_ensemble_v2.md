# mirfak_iota — ensemble v2: `<metric>0.21987<\metric>` (kNN-gated, improved kNN config)

Refs: [[mirfak_iota_knn_gate.md]] · [[mirfak_iota_ensemble_v1.md]] · [[betelgeuse_mu_cv_and_leakage_finding.md]] · [[castor_zeta_huber_big_gru.md]]

## Improvements over v1
1. kNN config tuned: similarity = **0.7·id(first-68) + 0.3·id(full-107)**, k=5, weight=sim^4. Better than id68-only.
2. Gate: `w = clip((sim_max − 0.8)/0.2, 0, 1)·0.3`, `p = (1−w)·NN_ens + w·kNN`. Split-half (30×): gain **+0.00279 ± 0.00039, positive 30/30**.
3. clip predictions at 0.
- kNN alone still weak (0.373), but gated blend: **simple NNLS 0.22273 → `<metric>0.21987<\metric>`** cols [0.2027, 0.2477, 0.2093].
- Gate activates partially on **165/240 test rows** (sim≥0.8), 124 rows ≥0.9.

**`final_solution/submission.csv` = ensemble v2.** Test array: `workspaces/mirfak_iota/ens2_test.npy`.

## Reusable artifacts (updated)
- `knn_best_oof.npy (2160,68,5)`, `knn_best_test.npy (240,107,5)`, `knn_best_S_tr.npy`, `knn_best_S_te.npy` — the chosen kNN + mixed-similarity matrices.
- `ensemble.py` now has `--knn-gate --clip 0`; per-col NNLS + gated kNN fully automated.

## Still queued
- My `g512so_snw` 5-fold ×2 seeds (fold sn1 ≈0.230), `gso_snw`, `lstm512so_snw` — ETA ~1h.
- Waiting on castor's hubergru 5-fold + test arrays (diverse: SN1-only, Huber, big GRU).
- I'll rerun the same recipe with the enlarged model pool; keep sending arrays.

#ensembling #status #submission #insight
