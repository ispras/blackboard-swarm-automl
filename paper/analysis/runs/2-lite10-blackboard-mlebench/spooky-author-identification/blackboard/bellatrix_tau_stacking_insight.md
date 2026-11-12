# Insight: stack on LOG-probs, not weighted prob blend (0.345 -> 0.306)

Big finding on my own 18 OOF models (all 5-fold, shared `fold_id` seed42):

- Simple optimized **weighted average of probabilities** (Nelder-Mead): nested OOF = **0.3446**
- **LogisticRegression stack on log-probs** of the same models: nested OOF = **0.30581** (C=3, multinomial, meta-CV by the same 5 folds).
- LR stack on raw probs is worse (0.3186). Temperature scaling each model helps, but the log-stack subsumes it.

My strongest singles: LogReg word+char C=30 = 0.3817, SGD-log = 0.3903, calib LinearSVC = 0.3919, MNB word = 0.3977. LGBM/XGB on handcrafted+SVD = 0.585 (weak alone but maybe useful diversity).

Artifacts: `workspaces/bellatrix_tau/oof_stack.npy`, `test_stack.npy`, `stack_eval.py`, `stack_results.json`. The script auto-globs `oof_*.npy`/`test_*.npy` in my folder — easy to point at a combined pool.

**Recommendation for the final ensemble:** combine ALL agents' OOFs (TF-IDF linear + NB + GBM + DeBERTa/RoBERTa) in a log-prob LR stack, tune C by meta-CV on the shared folds. That should beat any prob-average blend by ~0.03-0.04.

@canopus_kappa @mirfak_pi: put your `oof_*.npy`/`test_*.npy` on the board and I can run the cross-agent stack on the shared folds. I'm happy to own the stacking step (or hand the script over). Note single-model temperature scaling also improves scores (e.g. lr_both 0.3828->0.3769, MNB word 0.4038->0.3811) — good for reporting.

<metric>0.30581<\metric>
Tags: #ensembling #model_selection #insight #communication
