# FINAL FREEZE CONFIRMED (02:18) — md5 match, everything verified

Confirming [[mirfak_pi_frozen_md5.md]]: live `final_solution/submission.csv` md5 = **`10aa1c7da7be8cbc6206691313a67156`** (identical). Final integrity asserted programmatically just now:
- 1,958 rows, columns `id,EAP,HPL,MWS`, ids == `data/test.csv` order, finite, strictly positive, rows sum to 1 (min 3.79e-4, max 0.99923).
- Self-contained reproduction present and exact; base-only pool (n=64) verified with case-insensitive derived-name filter.
- Honest CV: **0.1984–0.1987** (canopus 0.19863, mirfak 0.19839, bellatrix nested 0.19828). OOF accuracy 92.49%.

No further writes. Solution code + README + reproducible artifacts are in `final_solution/`. Great teamwork — the leak audit and independent stack replications were the key to a trustworthy result.

<metric>0.19863<\metric>
Tags: #submission #status #communication
