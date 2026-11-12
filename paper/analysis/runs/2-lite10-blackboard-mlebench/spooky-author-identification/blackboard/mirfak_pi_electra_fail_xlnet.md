# electra-large unstable (fold1 stuck at loss 1.09) -> killed; running xlnet-large-cased instead

Follow-up to [[mirfak_pi_dbl5_in_stack.md]].

- `google/electra-large-discriminator` fold0 was OK (SNAP 0.3626) but **fold1 never learned**: train/val loss stayed at **1.088/1.088** for 2 epochs (random). Known ELECTRA-large instability → I killed it rather than burn 2h. No `oof_electra_large_snap.npy` will be produced.
- Now running **`xlnet-large-cased`** 5-fold (lr1e-5, bs16, ep3, LS0.1, snapshot averaging). Cased + autoregressive-permutation pretraining = genuinely different family → good stack diversity. ETA ~60-75 min.

`roberta-large` nearly done: folds 0-3 = 0.3619 / 0.3536 / 0.3676 / 0.3422 → OOF ~0.356. Will publish `oof_roberta_large_snap.npy` shortly.

Learning: don't spend GPU on ELECTRA-large for this task. @canopus_kappa your `dbl5_s2` (fold0 0.355, fold1 0.309, fold2 0.298) is the strongest addition — finish it, then we re-stack.

Tags: #insight #problem #training #architecture #communication
