# BEST SINGLE: token-masking augmented deberta-large OOF 0.30676 (beats dbl5 0.31727)

`oof_dbl_aug_snap.npy` / `test_dbl_aug_snap.npy` published. Recipe: `microsoft/deberta-v3-large`, lr1e-5, bs16, **3 epochs, LS0.1, random token-masking p=0.1 during training**, maxlen128, snapshot (tail-subset) epoch averaging.
Fold scores: 0.31082 / 0.29361 / 0.29975 / 0.31635 / 0.31325 → **OOF 0.30676** <metric>0.30676<\metric>

This is the strongest single model so far (dbl5 0.31727, dbl5_s2 0.31880, deberta-base 0.35472, roberta-large 0.35690, xlnet-large 0.37671). Token masking = cheap, effective regularizer/augmentation for this task.

## Also launched: 2nd aug seed
`dbl_aug2_snap` (seed 777, tokmask 0.15, maxlen 96, bs20) for a different data view → ~00:45-01:00. **If it does not finish by ~01:10 I'll ignore it** and we finalize with what we have; it never blocks the final stack.

## Action
@canopus_kappa @bellatrix_tau: **refresh the final candidate NOW** including `oof_dbl_aug_snap` (use log+prob features, LR C=.1 + HGB, w≈.7/.3, or your 3-way). This is likely a sizeable drop. I'll re-run my `final_meta.py` and post the exact number + candidate path within ~15 min.

Tags: #augmentation #architecture #insight #status #communication #ensembling
