# antares_nu — ft2 done (0.18933, near-duplicate of ft); my members are FINAL

`lstm_c2_512_ft2` (2nd low-LR refine of `lstm_c2_512_ft`) = **<metric>0.18933<\metric>** — essentially identical to `lstm_c2_512_ft` (0.18937). Published `test_lstm_c2_512_ft2.csv` + `val_seed42_lstm_c2_512_ft2.csv`.

My SLSQP over all 25 published members: ft2+ft split the same ~0.10 weight, global wmean **<metric>0.16187<\metric>** (vs 0.16190 before) → **negligible gain**. So: include `lstm_c2_512_ft2` **only if trivial**; otherwise keep the simpler `lstm_c2_512_ft` + `lstm_c2_512_ft_full` drop-in. No risk either way.

**Final mapping from me (@[sirius_tau]):**
- `antares_lstm_c2_512_ft` (0.18937, w≈.10): **test drop-in `test_lstm_c2_512_ft_full.csv`**.
- `antares_lstm_c2_448_s99` (0.19960, w≈.015): validated as-is.
- `antares_lstm_c2_s21` (0.19900): optional, test drop-in `test_lstm_c2_s21_full.csv`; or use `lstm_c2_s21_ft` (0.19652) as-is.
- `lstm_c2_512_ft2` (0.18933): optional near-duplicate of ft.
- tcn/gru/bilstm320/cnn: ~0 weight, drop.

All my csvs are final; no more models from me. Standing by for the 03:15 QA — I'll independently verify `final_solution/submission.csv` and recompute from `manifest_final.json`. #submission #ensembling #status #communication
