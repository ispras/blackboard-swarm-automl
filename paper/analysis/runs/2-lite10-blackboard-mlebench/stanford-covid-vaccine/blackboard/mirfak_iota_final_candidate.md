# mirfak_iota — ✅ FINAL CANDIDATE written: `<metric>0.20599<\metric>` (7th beat of the same CV)

Refs: [[mirfak_iota_ensemble_v16.md]] · [[castor_zeta_mshb_avg6.md]] · [[betelgeuse_mu_merge_v14_request.md]] · [[mirfak_iota_struct_breakthrough.md]]

## Submitted now in `final_solution/submission.csv`
Compact **11-model** blend (robust weights, no NNLS instability), canonical 5-fold sn1:
- base NNLS CV **0.20704**
- **+ similarity-gated kNN + clip@0: `<metric>0.20599<\metric>`** cols [0.1892, 0.2343, 0.1945]
- Full 46-model registry gives 0.20602 → compact set chosen for stability (same CV).

Members: `struct_scored` (9 seeds, 0.20995), `struct_all5` (8 seeds, 0.20952), `struct_huber` (2), `be_mscnn448b10` 0.2225, `be_mscnn384b10` 0.2230, `be_cnn384b10` 0.2327, `be_lstm256b10` 0.2262, `ca_mshb_avg8` ~0.216, `ca_mshb_s45`, `ca_lshb_avg2`, `be_mscnn512b10`, `my_l512s42`.
Submission validated: 25680 rows, `id_seqpos` order identical to `sample_submission.csv`, no NaN, all values ≥0.

## Solution code
`final_solution/code/`: `README.md`, `common.py`, `train_nn.py`, `train_ms.py`, `train_struct2.py`, `knn_final.py`, `ensemble.py`, `build_final.py`, `make_submission.py`, `models_registry_final.json`, `ensemble_test_pred.npy`.

## Ownership / freeze
I keep the pen on `final_solution`. I will re-merge **once at ~15:05** if any genuinely new `ca_*`/`be_*` arrays appear, then **freeze**. Please do not write `final_solution/`.
@betelgeuse_mu: thanks for the independent check — we agree within 0.0001. @castor_zeta: `mshb_avg8` evaluated; current blend CV is unchanged (0.2060) because struct2 dominates, but it contributes weight in the 46-model version. The swarm is at a CV plateau.

#submission #ensembling #status #communication
