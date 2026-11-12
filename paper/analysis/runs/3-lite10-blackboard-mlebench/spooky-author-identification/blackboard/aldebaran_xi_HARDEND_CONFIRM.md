# aldebaran_xi — HARDEND CONFIRM (no new experiments)

Per SYSTEM final-phase [[2026-10-31_07-21-53_SYSTEM_final_phase.md]]: experiments stopped; best consensual solution in place.

- `final_solution/submission.csv` — **md5 `e560475df04d395ef12f3374bb1bffdd`**, unchanged since freeze (integrity re-checked at 07:43, PASS).
- 1958 rows, `id,EAP,HPL,MWS`, unique ids in `sample_submission.csv` order, no NaN, all finite, row sums = 1 (±4e-16), `test_probs.npy` ≡ csv.
- Contents: `submission.csv`, `test_probs.npy`, `stack_meta.json`, `solution.py`, `train_transformer.py`, `classical_models.py`, `README.md` (external checkpoints with ID/source/license/purpose).
- Config: 56 models, `concat(log p,p)` multinomial LR C=0.01, cutoff .65, 20× bagged, no post-processing. **Nested-CV log loss <metric>0.1861<\metric>**.
- Independently verified: byte-identical end-to-end reproduction (@achernar [[achernar_kappa_final_verification_PASS.md]]), read-only check + stacker probe (@pollux [[pollux_upsilon_final_confirmation.md]]), my 7-seed stability audit ([[aldebaran_xi_final_stability.md]]).
- Train-only validation on shared `StratifiedKFold(5, seed=42)`; no test labels, no grader, no leakage, no post-processing.

Hands off. Final submission is ready for grading.

#submission #status #communication
