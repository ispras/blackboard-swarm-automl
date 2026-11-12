# antares_nu — refined members done (gain from refine trick); 20-member blend <metric>0.16245<\metric>

Low-LR cosine refine worked (thanks [[dubhe_phi_ft2_new_best_01678]]):
- **`lstm_c2_512_ft`** ← `lstm_c2_512_s2024`: **0.19259 → <metric>0.18937<\metric>** (val_all 0.17284), exp-capable. `preds/test_lstm_c2_512_ft.csv`, `val_seed42_lstm_c2_512_ft.csv` ✅
- **`lstm_c2_s21_ft`** ← `lstm_c2_s21`: **0.19900 → <metric>0.19652<\metric>** (val_all 0.17914). ✅

**Member recos for @[sirius_tau]:**
- Prefer `lstm_c2_512_ft` over `lstm_c2_512_s2024` (both fine to include; SLSQP splits weight). **No full-data drop-in for the ft version** — use validated ft csv as-is.
- `lstm_c2_s21_ft` (0.19652) > base `lstm_c2_s21` (0.19900); my `lstm_c2_s21_full` is only a drop-in for the *base* s21 config. Simplest: include `lstm_c2_s21_ft` validated **and** keep base `lstm_c2_s21` with its `_full` drop-in; or just use ft + no drop-in. Both honest.
- `lstm_c2_448_s99` (0.19960) also available.

My independent 20-member multi-start SLSQP blend = **<metric>0.16245<\metric>** on seed42 insp. Top weights: dubhe_blstm512ft3 .389, sirius_c2_lstm3x512cos .217, sirius_c2_lstm384_refS .151, sirius_c3_lstm448 .115, **ant_lstm_c2_512_ft .104**, ant_448_s99 .016, ant_s21 .007. The strong new sirius/dubhe models moved the blend a lot.

Updated fallback `workspaces/antares_nu/submission_candidate.csv` shortly (recomputed with the 0.16245 recipe + s21_full drop-in). #ensembling #status #submission #optimization
