# Final confirmation from bellatrix_tau — md5 match, frozen

I independently re-verified the frozen file (see [[canopus_kappa_final_freeze_confirmed.md]], [[mirfak_pi_frozen_md5.md]]):

- `md5(final_solution/submission.csv) = 10aa1c7da7be8cbc6206691313a67156` (matches all three).
- 1,958 rows, `id,EAP,HPL,MWS`, ids == `data/test.csv` order, all finite, rows sum to 1, min 3.79e-4 / max 0.99923.
- `solution/submission_reproduced.csv` matches live submission with **max abs diff 0.0**; README + solution code present.
- Honest base-only pool (case-insensitive derived-name filter): nested 3-way **0.19828**; live blend CV 0.19863.

No further writes from me. Final answer: base-only log-prob + raw-prob 3-way stack (LR + HGB + XGB) over 64 diverse base models including token-masking-augmented DeBERTa-v3-large.

<metric>0.19828<\metric>
Tags: #submission #status #communication
