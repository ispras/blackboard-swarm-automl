# achernar_delta — last experiments (all negative → final unchanged)

Refs: [[achernar_delta_signoff.md]], [[capella_kappa_tta_negative.md]], [[arcturus_kappa_cvnxb_done.md]]

Ran final low-risk validatable experiments on the shared fold-0 holdout (checkpoints, no retraining):

| experiment | result |
|---|---|
| combiner: arithmetic vs geometric vs **median** vs trimmed vs pow-mean (p=0.5/2/3) | 0.9007-0.9015, all within noise; **arith kept** |
| rank-mean | broken/0.73 (rejected) |
| multi-crop (5-crop 1.25x) TTA vs center-crop, ConvNeXt-s f0 | 0.8889 vs 0.8895 (plain), 0.8985 vs 0.8964 (mixup), blend 0.8889/0.8993 -> **no gain** |

Consistent with capella's aspect-preserving-TTA and arcturus's multi-scale results: **single square view is best for these nets**. 
Adding ConvNeXt-s@512 as a member was tested earlier and **hurt** fold-0 (0.9007->0.8993), so also excluded.

**`final_solution/submission.csv` stays frozen** (11-member arith, fold-0 0.9012). No change.

#insight #inference #ensembling #postprocessing #status
