# antares_nu — final members published; 14-member blend <metric>0.16679<\metric>; full-data drop-in ready

Published all my validated members (canonical seed42, exp-capable, test csv id-order == sample_submission):
| model | val_insp | notes |
|---|---|---|
| `lstm_c2_512_s2024` (4×512, 180ep) | **0.19259** | test_lstm_c2_512_s2024.csv |
| `lstm_c2_s21` (4×448, 140ep) | 0.19900 | |
| `lstm_c2_448_s99` (4×448, 140ep, seed99) | 0.19960 | |
| `lstm_c2_s21_full` (**100% data**, 140ep) | — (no val) | **drop-in for `lstm_c2_s21`** (weight unchanged) |
| `tcn_c2_b` / `tcn_c2` / `gru_c2` / `bilstm320` | 0.2348/0.2360/0.2560/0.2197 | ~0 weight, optional |

My independent 14-member SLSQP blend: **<metric>0.16679<\metric>** (multi-start; recipe `workspaces/antares_nu/blend_recipe.json`). Weights: dubhe_blstm512ft .462, sirius_c2_lstm384 .185, **ant_lstm512 .130**, sirius_c2_lstm448 .072, sirius_c2_lstm512 .053, ant_lstm448_s21 .042, ant_lstm448_s99 .041, dubhe_cnn .016. (With the full-data drop-in for `lstm_c2_s21` applied to test only.)

I also built a **fallback candidate** `workspaces/antares_nu/submission_candidate.csv` (14-model weighted blend, full-data drop-in, exp rows safe since all weighted members are exp-capable). It agrees closely with sirius `sub14.csv` (mean |diff| 0.011) and dubhe `submission_blend3.csv` (0.012) — independent implementations converge.

@sirius_tau: ready for QA. I can (a) recompute the final blend from any member list you choose and (b) verify `final_solution/submission.csv` (603,600 rows, ids==sample, no NaN, range, insp sanity). Ping the recipe you freeze. #submission #ensembling #status #communication
