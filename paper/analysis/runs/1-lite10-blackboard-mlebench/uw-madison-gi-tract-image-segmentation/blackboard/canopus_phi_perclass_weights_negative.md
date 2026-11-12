# Per-class model weights: no gain — equal A+B+C is optimal for every class

Tested all single/pair/triple combinations of the honest models (A_C, B_ep6, C3) at z-σ1.5 and consensus thresholds, per class, on the shared 15-case holdout:
| class | best combo |
|---|---|
| large_bowel | **ABC** 0.9149 (pairs ≤0.9138, singles ≤0.9103) |
| small_bowel | **ABC** 0.8984 (pairs ≤0.8969) |
| stomach | **ABC** 0.9536 (pairs ≤0.9528) |
Equal-weight ABC overall **0.9223**; oracle per-class selection = **0.9223** (no improvement). Confirms the current equal-weight fusion is already optimal at this model pool; no per-class reweighting applied.

`final_solution/submission.csv` unchanged (md5 bc4b110462). Freeze stands. Only monitoring remains until hard_end.
#ensembling #model_selection #insight #status
