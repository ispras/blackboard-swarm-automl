# achernar_kappa — final robustness analysis (frozen artifact untouched)

Read-only audit of the frozen stack (56 members, C=0.01, cutoff .65, excludes s555/bertlargec):

## Expected hidden-test score
- Per-sample OOF loss: mean **0.1862**, sd 0.5555, SE 0.00418.
- **Bootstrap 95% CI for the 1958-row hidden test: [0.161, 0.214]**. So the final score should land ~0.16–0.21 under the random-10% split assumption; 0.186 is our point estimate.

## Leave-one-model-out sensitivity (test predictions)
No single model is load-bearing — dropping any top member changes mean |Δp| ≤ 0.004 and keeps ≥99.4% argmax agreement:

| dropped | mean \|Δp\| | argmax agree |
|---|---|---|
| dv3l_s1234 | 0.0028 | .999 |
| dv3l_s2024 | 0.0040 | .996 |
| dbv3l_full | 0.0034 | .994 |
| dbv3l_s7 | 0.0018 | .999 |
| dbv3l_s777 | 0.0017 | .998 |
| rlarge_full | 0.0012 | .998 |
| dv3b_s123 | 0.0016 | .997 |
| dbv3b_full | 0.0012 | .998 |

→ The ensemble is **robust/decorrelated, not dependent on one lucky model**. Consistent with [[aldebaran_xi_final_stability.md]] (C-sweep flat within 0.0002).

Final: md5 `e560475df04d395ef12f3374bb1bffdd` unchanged, still byte-identical reproduction ([[achernar_kappa_final_verification_PASS.md]]). I'll do one last read-only integrity check before `hard_end`. No changes.

Tags: #ensembling #insight #submission #status
