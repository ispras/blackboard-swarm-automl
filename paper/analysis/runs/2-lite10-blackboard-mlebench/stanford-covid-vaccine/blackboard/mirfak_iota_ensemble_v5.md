# mirfak_iota — ensemble v5: `<metric>0.21510<\metric>` (17 models incl. castor hubergru)

Refs: [[mirfak_iota_ensemble_v4.md]] · [[betelgeuse_mu_new_models_blend.md]] · [[castor_zeta_huber_big_gru.md]] · [[mirfak_iota_knn_gate.md]]

## Result
Pulled betelgeuse's 14 full-107 OOFs + castor's `hubergru_avg2` (0.23347, seed-avg of 2) + my 2 GRUs (17 total):
- NNLS per-col CV = 0.21654
- **+ gated kNN = `<metric>0.21510<\metric>`** cols [0.1983, 0.2429, 0.2041] (v4 was 0.21576).
- `final_solution/submission.csv` = v5.

## Nonzero NNLS weights (col0 / col1 / col3)
| model | w |
|---|---|
| my_g512s42 (scored-only+SN-weight GRU) | 0.30 / 0.32 / 0.34 |
| ca_hubergru_avg2 (SN1+Huber big GRU) | **0.14 / 0.14 / 0.17** |
| be_mscnn384b | 0.10 / 0.13 / 0.13 |
| be_mscnn512 | 0.10 / 0.14 / 0.14 |
| be_cnn384 | 0.12 / 0.10 / 0.12 |
| be_lstm384 / cnn256 / mscnn256 / tf192 | 0–0.11 |
| all other LSTMs, bigru, lstm_sn1 | 0 |
Key: **three independent recipes** carry the blend — scored-only+SN-weight (mine), all-5+SN1-only+Huber (castor), all-5 MSE (betelgeuse). Diversity > single-model CV.

## Next
- Add my seed-2 / g256 / lstm512 as they finish (~30 min).
- Try castor `s42`+`s43` individually vs `avg2`.
- Betelgeuse: if a new model <0.226 appears, send it; otherwise we're in good shape.
- Freeze 17:15; I keep `final_solution` updated at each improvement.

#ensembling #status #submission #insight
