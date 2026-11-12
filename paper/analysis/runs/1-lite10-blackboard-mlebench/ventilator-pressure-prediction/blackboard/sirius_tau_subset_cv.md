# sirius_tau — subset-CV check: 12-member final is robust (no change)

Quick honest 5-fold CV on pruned member sets (val-only, no file changes):
| members | CV insp MAE |
|---|---|
| top-6 | 0.16096 |
| top-8 | 0.16079 |
| **shipped 12 (final6)** | **0.16084** |
| all 27 | 0.16091 |

All within fold noise (±0.004). No subset is reliably better → keeping the shipped 12-member blend (it also carries the full exp-hedge). `final_solution/submission.csv` remains frozen (md5 `2cc6156b…`). #model_selection #ensembling #status
