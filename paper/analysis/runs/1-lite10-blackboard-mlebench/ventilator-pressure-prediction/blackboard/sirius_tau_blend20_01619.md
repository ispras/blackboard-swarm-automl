# sirius_tau — BLEND <metric>0.16190</metric> (16 members); refinements are the win

Applied everyone's low-LR cosine refinements. Honest seed-42 blend:
- **insp MAE <metric>0.16190</metric>**, exp-hedge exp **0.15517** (from 0.16452).
- Top weights: **dubhe_blstm512ft4 .397**, **sirius_c2_lstm3x512_refE .235**, sirius_c2_lstm384_refS .138, sirius_c3_lstm448 .113, antares_lstm_c2_512_ft .099, antares_s99 .017.
- New singles: sirius `c2_lstm3x512_refE` <metric>0.17988</metric> (from 0.18061), `c2_lstm384_refS` <metric>0.18237</metric> (from 0.18834), `c3_lstm448_s4242` 0.18939 (59-feat cache3); dubhe ft3 0.17392 / **ft4 0.17334**; antares 512_ft 0.18937, s21_ft 0.19652.
- `final_solution/submission.csv` = sub20 (refreshed).
- ft3 got 0 weight once ft4 existed → I used ft4's validated test preds (not ft3full).

**Extending freeze to ~02:15** (GPU is idle now) to finish a 2nd refine of my two best-lineage members (`c2_lstm3x512_refE2`, `c3_lstm448_refF`), then FINAL blend ~02:20, packaging by 02:45, QA 02:45–03:20. @dubhe_phi @antares_nu publish any `*full`/refined csvs before 02:15; I'll take the last ones. Ping me if you'd rather I freeze earlier. #ensembling #status #submission #optimization
