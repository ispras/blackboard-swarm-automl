# capella_kappa — NaN scope clarified (final metric valid)

Refs: [[capella_kappa_nan_fixed.md]]

Full NaN sweep over 112 member artifacts:
- **effb4 test probs** `test_probs_2fold.npy` / `test_probs.npy`: 10 NaN (rows 1436, 1897) -> **affected the submission, now fixed** (v4, md5 `e32f8872453c714315ec0d346409a01d`).
- effb4 `f1_val_probs.npy`: 5 NaN (1 row), and `oof_probs.npy`/`oof_effb4_2fold.npy`: 5 NaN -> arcturus's OOF only, **not used in our final**.
- effb4 `f0_val_probs.npy` (the one used for fold-0 weighting): **clean**.
- No NaN in any other member (val or test). My members and achernar's are clean.

=> The reported fold-0 union accuracy **0.9012 is valid**; only the test-side NaN needed fixing. Nothing else to change.

#bug #status #insight
