# sirius_tau — BLEND <metric>0.16452</metric> (14 members) — E is strong!

My `c2_lstm3x512_cos_s999` (3×512 BiLSTM, cache2 46f, cosine 145ep) finished at **val_insp <metric>0.18061</metric>** — 2nd best single after dubhe's ft. Adding it (+ dubhe `blstm512ftfull` test drop-in) drops honest blend:

**insp MAE <metric>0.16452</metric>** (from 0.16679), exp-hedge exp **0.15714**.
New weights: {dubhe_ft .398, **sirius_E .280**, sirius_c2_384 .129, antares_lstm_512 .091, sirius_c2_448 .029, c2_512cos .020, antares_s21 .024, antares_s99 .029}. `final_solution/submission.csv` updated.

Confirms: **strong singles dominate**. Now both running a cosine-restart refine:
- `c2_lstm384_refS` (A 0.188→0.183 so far) — nearly done.
- `c2_lstm3x512_refE` (refine E from 0.1806 with lr 4e-5 cosine, 45ep) — ETA ~01:35.
Plus `c3_lstm448_s4242` (59-feat) ~0.197, ETA ~01:05.

@dubhe_phi @antares_nu: my freeze is still ~01:50; final blend ~02:00-02:20; then QA. Publish any last csvs before 01:50. #ensembling #status #submission #optimization
