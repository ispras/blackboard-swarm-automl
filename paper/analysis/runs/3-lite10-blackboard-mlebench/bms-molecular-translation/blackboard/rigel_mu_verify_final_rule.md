# rigel_mu — INDEPENDENT 20k verification of the final rule: canon(rigel)+high-conf-1m = <metric>7.713–7.760<\metric>

Refs: [[altair_epsilon_canon_rule.md]] [[alderamin_xi_final_rule_085.md]] [[rigel_mu_rigel_everywhere.md]]

I recomputed everything independently on the **full 20,000-row holdout** (my `rigel_val20k.csv` + your 1m 20k run `/tmp/opencode/h20k/s{0,1}.csv` converted with altair's `smi_to_inchi_fallback` + RDKit):

| rule | mean Lev (20k) |
|---|---|
| 1m alone | 40.259 |
| rigel raw | 8.583 |
| **canon(rigel)** (RDKit round-trip only when parses; 295 rows changed) | **8.363** |
| canon + 1m if valid & score≥0.80 (n=6597) | 8.148 |
| **canon + 1m if valid & score≥0.85 (n=4555)** | **7.760** |
| **canon + 1m if valid & score≥0.875 (n=3040)** | **7.713** |
| canon + 1m if valid & score≥0.88 (n=2688) | 7.746 |
| canon + 1m if valid & score≥0.90 (n=1075) | 7.993 |
| raw (no canon) + 1m if valid & score≥0.85 | 7.941 |

**Confirms both of your findings independently.** Canonicalization is safe (-0.22) and the high-confidence 1m blend adds another -0.6.
- **Recommended threshold: T=0.85** (7.760) or **0.86–0.875** (≈7.71–7.72). The 0.80–0.90 range is monotone-ish and stable; 0.875 is the empirical min but 0.85 is more conservative. Either is fine; I'd ship **0.85** for robustness (0.05 Lev difference vs 0.875).
- **Order of operations (important):** `r = canon(rigel)` first, then `if 1m_valid and 1m_score>=T: 1m else r`. 1m strings are already canonical.

## Test application
`preds1m_inchi.csv` has `inchi` + `valid` for all 484,838; `rigel_test_all.csv` will be complete ~17:25. So the rule is directly applicable. Please make sure the final writer canonicalizes rigel on **all** rows where `MolFromInchi(s, sanitize=False, removeHs=False)` succeeds (keep raw otherwise), and applies the 1m override only with `valid & score≥0.85`.

When `final_solution/submission.csv` is written, ping me — I'll run my `validate_submission.py` + recompute the diff vs pure rigel.

Tags: #ensembling #postprocessing #model_selection #status #submission
