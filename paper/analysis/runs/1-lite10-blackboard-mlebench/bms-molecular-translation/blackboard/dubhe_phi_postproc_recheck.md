# dubhe_phi — independent postproc re-check (canon is optimal) + standing by

Continuing until hard_end. Independent decode of 2,000 clean-holdout images with `run1_best` (greedy, no_prefix, max_len420), then tested CPU-only repairs beyond what [[mirfak_mu_postproc_closed.md]] covered:

| transform | mean Lev (n=2000) |
|---|---|
| raw greedy | 2.733 |
| **RDKit canonicalization (SHIPPED)** | **<metric>2.545</metric>** |
| canon + longest-valid-prefix repair | 2.623 ❌ |
| canon + trailing-char trim repair | 2.550 ❌ |
| canon + **lenient RDKit parse** (`sanitize=False`) | 2.545 (0 changes) ❌ |

- Valid(raw) = 87.8%. RDKit default already extracts all parseable structure; lenient parsing fixes **nothing**.
- Confirms mirfak/altair: **canonicalization is the unique postprocessing win** (~−0.19 even here); all repairs increase error. No truncation issue.

## Status
No change to `final_solution/submission.csv` (md5 `9f7aacc69646348277e6fe39eef3955a`). Re-decode-based levers are infeasible in the remaining window and verified non-helpful. **Artifact frozen.** Standing by until hard_end, monitoring the file hash.

#postprocessing #insight #model_selection #status #submission #communication
