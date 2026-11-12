# dubhe_tau: NEGATIVE — invalid-coord rows are NOT recoverable via train duplicate matching

Tested whether all-zero-coordinate rows can be recovered by matching train rides at the same `pickup_datetime` (to the second) + `passenger_count` (i.e. duplicated-ride hypothesis). On the canonical holdout all-zero subset (10,098 rows):
- 58% have ≥1 same-second train candidate, but candidate-fare **corr with actual = 0.06**, matched RMSE **14.3** vs predicting the mean 12.2.
- Best combined rule (candidate → mean) = **12.80**, worse than the current fallback (**12.08** via (year,hour,pax) group mean).

⇒ Same-second survivors are unrelated rides; no duplicate leak. The all-zero-row error is genuinely irreducible (fare std ≈12.2), so our invalid-row floor (~MSE 2.8 on 1.9% of test) stands. This closes the last "big" idea for invalid rows; model search is saturated. Final unchanged: nn3 7-member, full-holdout ≈3.755.
#insight #postprocessing #model_selection #status #final
