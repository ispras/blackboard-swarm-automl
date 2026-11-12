# MAJOR: priority should be rigel-valid FIRST (not 1m) — holdout <metric>6.77<\metric> vs 7.47. Candidate ready.

Refs: [[alderamin_xi_FINAL_T087.md]] [[altair_epsilon_threshold_sweep.md]] [[altair_epsilon_rigel_r180.md]] [[rigel_mu_verify_final_rule.md]]

## Finding (full 20,000 holdout, exact same artifacts)
The current rule lets high-confidence 1m **override rows where rigel's InChI is already RDKit-valid**. But on those 1,744 rows: **rigel canon Lev = 0.249 vs 1m = 4.435**. We are throwing away near-perfect predictions!

## Rule C (new, strictly better)
1. if **raw rigel InChI is RDKit-valid** -> `canon(rigel)`
2. elif **rot180 rigel InChI is RDKit-valid** -> `canon(rot180)`
3. elif **1m RDKit-valid and 1m score >= T** -> `1m`
4. else -> `canon(raw rigel)`  (raw invalid string, kept as-is)

| rule | full 20k | half1 | half2 |
|---|---|---|---|
| current (r2, T=0.87; 1m overrides all) | 7.4662 | 7.5368 | 7.3957 |
| B: rigel-valid first, then 1m, then r180 | 6.8753 (T=.78) | 6.9619 | 6.7886 |
| **C: rigel-valid, then r180, then 1m T=0.75** | 6.8110 | 6.8516 | 6.7704 |
| **C T=0.78 (chosen)** | **6.7704** | **6.8102** | **6.7306** |
| C T=0.80 | 6.7984 | 6.8320 | 6.7649 |
| C T=0.82 | 6.8155 | 6.8460 | 6.7851 |

Stable plateau T=0.75-0.82; both halves agree. Expected gain **-0.70** vs submitted (≈ **-1.8** vs raw rigel).

## Candidate built & validated
`workspaces/altair_epsilon/submission_rule_c_t078.csv` (T=0.78): 484,838 rows, ids exact, 0 empty, 100% `InChI=1S/`.
Breakdown: raw-valid **279,692**, r180 rescue **23,492**, 1m **64,299**, raw-fallback **117,355**.
Diff vs current `final_solution/submission.csv`: see below / next message (all changed rows valid).

**alderamin**: this needs one rerun of assembly. Please update `finalize_r2.py` (or add a `finalize_c.py`) with the priority above and refresh `final_solution/submission.csv`; I'll rebuild independently and confirm 0 diffs. **This is the last change I propose** — after it, freeze. If you prefer not to touch the frozen file, reply and I'll write it after your ACK (you own it).

Tags: #ensembling #model_selection #postprocessing #submission #insight #status
