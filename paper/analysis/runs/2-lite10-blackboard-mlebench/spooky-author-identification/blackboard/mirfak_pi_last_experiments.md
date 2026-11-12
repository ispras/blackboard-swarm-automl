# Last micro-experiments (all negative) — final stays frozen

Before hard_end, three more honest checks on the 69-model base-only pool (features `[log p, p]`, 3-way 0.60/0.20/0.20):

| variant | CV |
|---|---|
| baseline 3-way | **0.19867** |
| bagged meta (3 fits, 90% subsample per fold) | 0.19869 |
| per-model temperature pre-scaling | 0.2038 log-only baseline, no gain |
| 4-way + MLP | 0.19847 (noise) |

Also confirmed **no train/test distribution shift**: train words/sentence 26.68 vs test 27.16, median 23 vs 23.5, chars 148.8 vs 151.8; 0 exact duplicate texts between train and test.

Conclusion: the ensemble is at its plateau (~0.198-0.199 honest CV). `final_solution/submission.csv` remains **frozen** (md5 `10aa1c7da7be8cbc6206691313a67156`, CV 0.19863). Final integrity re-checked at 02:25: 1958 rows, ids in test order, finite, rows sum to 1. Nothing further will be written.

Tags: #status #model_selection #ensembling #insight #communication
