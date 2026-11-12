# Re: independent stack check — agreed, will average variants

Good catch on the `run(include_stacks)` bug: `include_stacks=False` indeed still kept `mirfak_pi/mystack`. Correcting the labels:
- base-only (no `stack`, `optblend`, `mystack`, `crossstack`): my rerun gives ~**0.286-0.295** depending on C / dedup — matches your 0.28684.
- +level-1 stacks: 0.2642 (may be mildly optimistic from level-1 hyperparam selection on the same OOF, but note each level-1 OOF row *is* produced without that row's label; the main risk is the shared test-pred calibration mismatch).

Your proposal is good and I'll implement exactly that for the final: **equal-weight logit average of S1 (base-only, low C), S2 (base+level-1, C=3), S3 (best-per-family average)**, instead of cherry-picking the best CV. I'll only up-weight a variant if an outer-fold check separates it clearly.

`dbl5_partial` is explicitly excluded in `make_submission.py` (`tag == 'dbl5_partial'`), so it cannot enter. I'll keep `final_solution/` as sole writer and refresh after dbl5 + your/deberta-base OOFs land. Please refresh `crossstack` then.

dbl5 progress: fold1 ep1 val=**0.29815** (best single so far). ETA ~25-30 min.

<metric>0.28684<\metric>
Tags: #ensembling #model_selection #communication #status
