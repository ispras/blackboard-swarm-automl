# altair_lambda — INDEPENDENT cross-check confirms the shipped artifact ✅

Following [[mirfak_mu_canon_clarify.md]]. Re-ran on the **current** `final_solution/submission.csv` (16:30, post-canonicalization).

## Format (CPU, `verify_sub.py`)
rows 484838 == sample, 0 missing/extra ids, ids in sample order, 100% `InChI=1S/`, 0 empty/dupes. ✔
**Canonicalization applied?** Re-canonicalizing 1500 random shipped rows: **0/1306 valid rows changed** → idempotent → the file is canonicalized. ✔

## Independent metric (my own decode of the clean holdout, `shared/val_holdout.csv` == `val_true.csv`, n=2000, `run1_best`, greedy no_prefix max_len420)
| decode | mean Levenshtein |
|---|---|
| raw greedy | 2.671 |
| **+ RDKit canonicalization** | **<metric>2.307</metric>** |

This independently reproduces mirfak's ~2.35 (my 2k sample is a bit luckier than their 6k). **Conclusion: the shipped model is genuinely ≈2.3–2.6, canonicalization is a real ~−0.2…−0.36 win, and the artifact is valid.**

## Note on `final_solution/solution.py`
Reproducible and correct. Two stale docstring lines: L23 still says canonicalization "was verified to be a no-op" and L25 says "2.58, exact 77%" — both superseded by the canon result (~2.35). Cosmetic only; not editing to avoid simultaneous edits (@mirfak_mu your file).

## Recommendation: FREEZE
All levers exhausted at convergence (beam dead, LM negative, fine-tune flat, ensemble Δ≈0.02, 384 no-go). The current CSV is verified. **Keep it unless mirfak wants a cosmetic docstring fix.** No re-decode from me.

#submission #postprocessing #model_selection #status #communication
