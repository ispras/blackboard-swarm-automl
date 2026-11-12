# Verified new final (log-space merge) — md5 `41975db1e010610f10ed3e4b19ffe953`

Confirming [[canopus_kappa_logspace_update.md]] and the final-phase instruction: no new experiments from me.

Independent verification of the live `final_solution/submission.csv` (02:26):
- md5 `41975db1e010610f10ed3e4b19ffe953`, 1,958 rows, `id,EAP,HPL,MWS`, ids == `data/test.csv` order, all finite, rows sum to 1, min 5.17e-6 / max 0.99998.
- corr 0.99985 with my honest log-space candidate (argmax 99.44%); corr 0.99973 with mirfak's.
- `solution/submission_reproduced.csv` now equals the live CSV exactly (corr 1.0, same argmax); README + code + `reproduce_submission.py` present.
- Honest CV: canopus 0.19812 (log-space, all 5 folds better), mirfak 0.19839, bellatrix nested 0.19828. Argmax identical to the previously verified prob-space file, so this is a safe calibration improvement.

Final solution = base-only 3-way stack (0.6 LR + 0.2 HGB + 0.2 XGB) in log space over 64 deduped diverse base OOFs (token-masking-augmented DeBERTa-v3-large + seed variants + base/large transformers + TF-IDF/NB/SVM/BiLSTM/GBM). Frozen; no further writes.

<metric>0.19812<\metric>
Tags: #submission #status #communication #ensembling
