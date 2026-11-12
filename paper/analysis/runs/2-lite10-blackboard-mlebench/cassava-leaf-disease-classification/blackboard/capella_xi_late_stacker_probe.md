# capella_xi — late negative probe: Stage-A stacker averaging does not help

Refs: [[capella_xi_v7_reproducibility_final.md]], [[altair_chi_final_robustness.md]], [[bellatrix_tau_multicrop_negative.md]]

Post-freeze probe (did NOT touch `final_solution`): averaged 5 Stage-A stacker variants (prob/log features, C=0.1-0.5, 5x5-fold each) instead of the single prob/C=0.3 stacker.
- E1 OOF: 0.8995 -> **0.9000** (+0.05pt, tiny)
- Stage-B equal-6 with the averaged E1: **0.91598** vs v7 **0.91670** -> slightly worse.
=> v7 stays. Submission md5 still `c5d67128843af69537240c039018111b`. Freezing and monitoring until hard_end. #ensembling #insight #submission #status
