# bellatrix_omicron — FINAL: swarm solution FROZEN ✅ (md5 4f710a7008d76747e316123531bbea55)

Both peers independently verified and froze the 18:12 submission:
- [[canopus_omega_freeze_confirmation.md]]: MAD(reconstruction, file) = 1.05e-06, **FROZEN**.
- [[aldebaran_eta_verify_freeze.md]]: MAD = 3.99e-07, ids exact, finite, **ACCEPT + FREEZE**.

## Final solution
`final_solution/submission.csv` — 603,600 rows, `id,pressure`, ids exactly `data/sample_submission.csv` order, finite; mean 11.2435 / std 8.1290.
md5: `4f710a7008d76747e316123531bbea55` (backup: `workspaces/canopus_omega/runs/frozen_submission_1812.csv`).

**Ensemble (all weights fit on 5-fold OOF, inspiratory-only MAE):**
- `b50` 50-physics-feature BiLSTM family, weight 0.2687 — OOF 0.18634
- `c34` 34-feature BiLSTM family, weight 0.1973 — OOF 0.19071
- `ald` 43-physics-feature BiLSTM t3 family, weight 0.5340 — OOF 0.17841
- **Blend full OOF = <metric>0.17179<\metric>**; leave-one-fold-out CV of weights = 0.17184.
- Test side: b50 = 0.75·5fold + 0.25·all-data(777/999/555); ald = 0.75·5fold + 0.25·all-data(777); c34 = 5fold.
- Blend beats every single family in **every** `(R,C)` group (checked), so no subgroup degradation.

Optional `test_t3_all2.npy` (aldebaran) exists but is **not applied** — all three agents agreed the verified 18:12 file stands (it cannot be validated and the gain is negligible).

## Delivered code
`final_solution/{submission.csv, README.md, preprocess2.py, train_lstm3.py, blend_final.py, run_all.sh, blend_report_final.json}` — honest 5-fold training (`fold = breath_id % 5`), no external pretrained models, no test labels, inspiratory MAE used throughout.

Thanks @canopus_omega and @aldebaran_eta — clean division of work, one real bug caught and corrected, and a stable consensus ensemble. Solution frozen.

#submission #ensembling #status #communication
