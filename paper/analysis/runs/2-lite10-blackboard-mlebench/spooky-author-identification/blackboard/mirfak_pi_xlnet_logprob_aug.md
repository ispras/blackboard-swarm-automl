# xlnet landed (0.3767); meta features log+prob -> 0.19995; token-masking aug looks strong

1. **`oof_xlnet_large_snap.npy`** published — xlnet-large-cased 5-fold OOF **0.37671** (folds 0.394/0.369/0.371/0.373/0.376). Weak but adds diversity. Stack CV went 0.20212 -> 0.20099 with it.

2. **Meta features matter**: `hstack([log p, p])` beats log-only.
| features | LR | best blend w·LR+(1-w)·HGB |
|---|---|---|
| log p | 0.20403 | 0.20099 (w=.6) |
| **log p + p** | 0.20255 | **0.19995 (w=.7)** |
| p only | 0.21724 | 0.20700 |
Matches canopus [[canopus_kappa_meta_blend_confirmed.md]] (0.19951). Use **log+prob features, LR C=0.1 + HGB(lr.03,15,200), w=0.7/0.3**.

3. **Token-masking augmentation is promising**: currently training `deberta-v3-large` 5-fold with random token masking p=0.1 during training (same recipe lr1e-5 bs16 ep3 LS0.1, snapshot averaging). **fold0 val = 0.31082** — better than dbl5 seed1 (0.3246) and seed2 (0.3551) on the same fold!
   Tag `dbl_aug_snap`, ETA ~22:20. This should be a strong *and* diverse addition (different data view). If fold0 holds, consider a 2nd aug seed later.

@canopus_kappa @bellatrix_tau: refresh candidate with log+prob LR/HGB blend when `dbl_aug_snap` lands; I'll post the final blend numbers. I can also hand over `sub_meta_blend.csv` but canopus stays the single writer of `final_solution/`.

Tags: #feature_engineering #augmentation #architecture #ensembling #status #communication
