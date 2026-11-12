# bellatrix_delta — last-mile combiner test: NO change justified (frozen stays)

Ran one more independent last-mile experiment on the FINAL upgraded member val-probs (nested CV inside fold0, 5 repeats × 5 folds):
| combiner | CV acc | vs uniform |
|---|---|---|
| arithmetic mean (frozen) | 0.89880 | — |
| accuracy-proportional weights | 0.89880 | +0.00000 ±0.0002 |
| logistic-regression stacking (8×5 log-probs, C=0.05) | 0.89607 | −0.0027 ±0.0010 |
(plus vega's earlier sweep: vote .89933, power/median/trimmed/geometric all ≤ .8988; temperature/class-prior/pseudo-label all negative.)
⇒ **No method beats the uniform average with statistical significance on the 3,745-row holdout.** Keep the frozen file.

`final_solution/submission.csv` unchanged: md5 `488eb8ecf72106b0342674cb7997d9ed`, 2676 rows, fold0 <metric>0.89880</metric>, independently verified 0/2676 twice ([[vega_eta_closing.md]] [[aldebaran_epsilon_FINAL_verified.md]]). Work complete; will only re-check integrity until hard_end.

#ensembling #insight #submission #status
