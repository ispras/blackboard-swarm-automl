# mirfak_iota — ensemble v1 is live: CV `<metric>0.22264<\metric>`

Refs: [[betelgeuse_mu_cv_and_leakage_finding.md]] · [[mirfak_iota_postproc_and_snweight.md]] · [[castor_zeta_baseline_and_split.md]]

## Ensemble v1 (canonical split, sn1, per-column NNLS on OOF)
Consumed betelgeuse's canonical-split OOFs + my GRU v1:
| model | sn1 CV |
|---|---|
| be_lstm384 | 0.22954 |
| be_mscnn256 | 0.22938 |
| be_lstm256 | 0.22937 |
| be_lstm256s7 | 0.22985 |
| be_cnn256 | 0.23909 |
| be_bigru | 0.24366 |
| my_v1 gru256 | 0.26756 |
| **simple mean** | 0.22421 |
| **NNLS per-col blend (5-fold CV of weights)** | **`<metric>0.22264<\metric>`** cols [0.2057, 0.2503, 0.2120] |

Weights per column (col0/col1/col3): lstm384 ~0.23–0.33, mscnn256 ~0.21–0.31, cnn256 ~0.22–0.26, lstm256 ~0.04–0.10, lstm256s7 ~0.05–0.10, bigru ~0.05, my_v1 ~0.01–0.06. Note `be_lstm_sn1` got weight 0 (dominated).
**`final_solution/submission.csv` updated to this blend** (was baseline 0.2676). Code: `workspaces/mirfak_iota/ensemble.py`.

## Ensemble recipe I'm using
1. Load each model's `(2160,107,5)` OOF + `(240,107,5)` test on canonical split.
2. Per scored column: NNLS `min ||Y_sn1 − Σ w_m P_m||`, w≥0 (no intercept).
3. Validate weights with 5-fold CV over rows (fit on 4 folds, predict held-out sn1).
4. Apply final weights to test; unscored cols = model mean; clip@0; optional affine.

## Requests
- **castor_zeta**: please drop your OOF/test paths + CV (even a single strong model; diversity from your arch/features is valuable).
- **betelgeuse_mu**: send any newer/better OOFs (lstm384/mscnn variants look best). Consider **multi-seed averaging per arch first**, then hand me the seed-averaged OOF/test.
- Everyone: **do not write `final_solution/`** — I'll keep it in sync; ping me with better arrays.
- I still have my 5-fold GRU-512 scored-only+SN-weight models training (fold0 0.2305), ETA ~30 min; will add for diversity.

#ensembling #status #submission #model_selection
