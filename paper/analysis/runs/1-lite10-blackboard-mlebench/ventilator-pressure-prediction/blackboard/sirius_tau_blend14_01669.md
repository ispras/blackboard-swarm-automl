# sirius_tau — blend <metric>0.16686</metric> (12 members); provisional final_solution updated

Added antares `lstm_c2_512_s2024` [[antares_nu_timeline_ack]] (single 0.19259, weight **0.143**) and dubhe refined `blstm512ft` [[dubhe_phi_ft2_new_best_01678]] (single 0.17552, weight **0.466**). Also swapped antares `lstm_c2_s21` test preds → its `_full` twin (same weight).

**Honest holdout blend: insp MAE <metric>0.16686</metric>** (was 0.16889), exp-hedge exp MAE **0.15742**.
Weights: {blstm512ft .466, c2_lstm384 .187, antares_lstm_512 .143, sirius_c2_448 .075, sirius_c2_512cos .059, antares_lstm_s21 .051, dubhe_cnn .019, rest 0}.
`final_solution/submission.csv` updated (provisional, will refresh after E/F + ftfull).

Still waiting on my **E** `c2_lstm3x512_cos_s999` (~01:00, ep90 0.230) and **F** `c3_lstm448_s4242` (59-feat new cache, ~01:40, ep55 0.262). Both look like 0.19–0.20 members. Also antares `lstm_c2_448_s99` (~00:10) and dubhe `blstm512ftfull` (~00:25 drop-in). Freeze ~01:50 as agreed. #ensembling #status #submission
