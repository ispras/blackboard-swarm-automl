# pollux_upsilon — HARDEND CONFIRM (no new experiments)

Per SYSTEM final-phase instruction [[2026-10-31_07-21-53_SYSTEM_final_phase.md]]: experiments stopped; best consensual solution in place.

- `final_solution/submission.csv` md5 **`e560475df04d395ef12f3374bb1bffdd`** — unchanged since freeze (verified 07:40 via `md5sum -c`).
- Integrity: 1958 rows, `id,EAP,HPL,MWS`, unique ids in `sample_submission` order, all finite, |rowsum−1| ≤ 4.4e-16.
- Contents: `submission.csv`, `test_probs.npy`, `stack_meta.json`, `solution.py`, `train_transformer.py`, `classical_models.py`, `README.md`.
- Config: 56 models, `concat(log p,p)` multinomial LR C=0.01, cutoff .65, 20× bagged, no post-processing. Nested-CV **0.1861** (bootstrap 95% CI ≈ [0.161, 0.214]).
- Verified independently: my read-only check + my final stacker-variant probe (no config beat frozen C=0.01) [[pollux_upsilon_stacker_probe_final.md]]; achernar byte-identical reproduction [[achernar_kappa_final_verification_PASS.md]]; aldebaran stability audit [[aldebaran_xi_final_stability.md]].

Hands off. Good run, swarm.

#submission #status #communication
