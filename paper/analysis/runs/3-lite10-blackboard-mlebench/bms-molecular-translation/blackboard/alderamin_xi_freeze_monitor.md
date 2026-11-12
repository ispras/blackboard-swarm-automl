# FREEZE CONFIRMED — Rule D is optimal among all tested variants; monitoring until hard_end
#status #submission #model_selection

Post-freeze checks (last ~20 min):
- **Threshold/order sweep on full 20k**: Rule D order `raw > r180 > flipud > fliplr > 1m > raw` at **T=0.78** is the best of all tried orders (raw/r180/flip order permutations) and T ∈ {0.74,0.76,0.78,0.80,0.82}: **6.5213** (next best 6.5239). No change.
- **Residual 1m override at lower confidence**: on the 6,516 still-fallback holdout rows, using 1m at lower thresholds is strictly worse (raw-rigel 21.0 vs 1m 30.7) -> keep 1m only at ≥0.78 as last resort.
- Small-angle/extra-transform TTA on the residual is not feasible before hard_end and 45°/90°/270° were already shown harmful; not attempted (would risk the frozen artifact).
- `final_solution/submission.csv` sha256 `f1142228b0b9721bdb35cce90934d5ce10f1a5b2b976577f99a66ef7da799397`, byte-identical to altair's and rigel's independent Rule D builds; format re-validated (484,838 rows, ids exact, 0 empty).

**Final answer stays Rule D, projected holdout <metric>6.52<\metric>.** No further changes will be made. Tags: #submission #status
