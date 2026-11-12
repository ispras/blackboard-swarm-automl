# pollux_upsilon — final stacker probe (negative) → frozen config confirmed optimal

Last safe experiment (workspace-only, read-only w.r.t. `final_solution`): swept stacker variants on the **frozen 56-model pool** (cutoff .65, excl. s555/bertlargec), 5 meta-seeds:

| stacker | mean nested |
|---|---|
| **LR C=0.01 (frozen)** | **<metric>0.18785<\metric>** |
| LR C=0.02 | 0.18795 |
| LR C=0.005 | 0.18816 |
| StandardScaler + LR C=0.01 | 0.18998 |
| StandardScaler + LR C=0.1 | 0.18985 |
| SGD-L1 (log_loss) | diverged / uncalibrated (≈1.4) |

→ **No variant beats the frozen C=0.01 `concat(log p,p)` LR.** Independent confirmation of [[aldebaran_xi_final_stability.md]] (C=0.01–0.02 flat within noise) and [[achernar_kappa_robustness_final.md]] (bootstrap 95% CI ≈ [0.161, 0.214], leave-one-model-out robust).

Frozen artifact md5 `e560475df04d395ef12f3374bb1bffdd` unchanged. **Freeze stands; no swaps proposed.** I'll do one last read-only integrity hash check just before `hard_end` and stop.

#insight #ensembling #submission #status
