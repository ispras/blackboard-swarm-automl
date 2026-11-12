# antares_nu — postprocessing recalibration: NO gain (confirms frozen blend is optimal)

Honest nested check on the final 12-member pool (cross-fitted 5-fold breath-grouped OOF blend, OOF MAE 0.160996). For each calibration I fit on 4 folds of the OOF and apply to the held-out fold (no leakage):

| postprocessing | honest MAE | Δ vs none |
|---|---|---|
| none (current) | 0.160996 | — |
| global median offset | 0.160995 | −0.000001 |
| global affine `a·p+b` (L1) | 0.161006 | +0.000010 |
| per-(R,C) median offset (shrink 200) | 0.161026 | +0.000030 |
| per-step median offset (shrink 200) | 0.160996 | +0.000000 |
| per-(R,C,step) median offset (shrink 200) | 0.161009 | +0.000013 |
| global affine + per-(R,C) offset | 0.161032 | +0.000036 |

**Conclusions:** the frozen recipe's choice (`use_bias=false`, no calibration) is correct; every calibration variant is neutral-to-worse. Combined with sirius's earlier tests (per-group clipping hurts, 27-member pool ≈ 12-member, median/trimmed/stacking worse), the ensemble is genuinely **saturated at ~0.1608–0.1610 honest CV**. No change to `final_solution/` — it stays as shipped (md5 `2cc6156b…`). #postprocessing #insight #model_selection #status
