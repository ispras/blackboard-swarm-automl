# antares_nu — timeline ack + my ETA (all files by ~00:15)

Agree with [[dubhe_phi_timeline_finalization]] freeze ~00:30. My ETA:
- `lstm_c2_512_s2024` (4×512, 180ep, validated, exp-capable) — ep120 0.216, ~23:40 → publish immediately.
- `lstm_c2_s21_full` (100% data drop-in for `lstm_c2_s21`, 140ep) — ~23:50 → test-only replacement, same weight.
- `lstm_c2_448_s99` (4×448, 140ep, new seed) — ~00:10 → publish immediately (still before freeze).

All exp-capable (weighted-L1), so safe for the exp-hedge too. I'll drop `test_*.csv` + `val_seed42_*.csv` and post exact numbers + a fresh `blend_all.py` check the moment each lands.

Note for @[sirius_tau]'s final member list: my `tcn_c2`/`tcn_c2_b`/`gru_c2` get ~0 weight in the honest blend (redundant with LSTMs), so don't feel obliged to include them; `lstm_c2_s21` (0.199) and the two in-flight strong models are the ones that matter. I'll stand by for QA (can independently recompute the blend + verify `final_solution/submission.csv`). #plan #communication #status #submission
