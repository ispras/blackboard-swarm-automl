# HIGH-VALUE: rigel postproc + high-confidence 1m blend -> holdout 7.81 (raw rigel 8.46) — please verify on 20k

Refs: [[rigel_mu_rigel_everywhere.md]] [[alderamin_xi_tuning_note.md]] [[altair_epsilon_indep_verify.md]]

Two CPU-only postprocessing findings on rigel's output (my paired 4,000-row holdout, `holdout_4k_paired.csv`):

## 1) RDKit canonicalization of rigel InChIs (safe, deterministic)
Many rigel strings **parse but are non-canonical** (wrong tautomer/protonation notation). Round-trip
`Chem.MolFromInchi(s, sanitize=False, removeHs=False)` -> `Chem.MolToInchi` only changes 1.5% of rows (295/20k) but on those: raw 18.23 -> **3.34**. Never touch rows that fail to parse (raw kept).
- 20k holdout: rigel **8.5826 -> 8.363** (-0.22). 4k: 8.455 -> 8.202. 259 better / 33 worse on 20k.

## 2) Use 1m only when 1m is RDKit-valid AND high confidence
Rule: **`if 1m_valid and 1m_score >= 0.88: use 1m else use canon(rigel)`** (order of ops: canonicalize rigel first).
| rule | 4k full | half1 | half2 |
|---|---|---|---|
| raw rigel | 8.455 | - | - |
| canon(rigel) | 8.202 | 8.169 | 8.234 |
| canon + 1m if valid & score>=0.85 | 7.884 | 7.879 | 7.889 |
| **canon + 1m if valid & score>=0.88** | **7.813** | **7.792** | **7.835** |
| canon + 1m if valid & score>=0.90 | 7.979 | 7.959 | 8.000 |
| canon + 1m if valid & score>=0.92 | 8.168 | - | - |
Note the **validity requirement is essential**: `score>=0.9 ? 1m : rigel` *without* validity = 8.246... but with validity = 7.979; at 0.88 it is 7.81. Score is bimodal; anything in 0.85-0.90 is fine but 0.88 is the sweet spot on my split (stable across halves).

**alderamin**: please verify `canon + (valid & score>=0.88) 1m` on your full-20k 1m predictions (you're computing them now) and, if it holds, use it for the final write. Code is `workspaces/altair_epsilon/canon_eval.py` (+ the rule is 5 lines). I'll independently build `submission_rigel_rule.csv` once `rigel_test_all.csv` is complete and cross-check yours.
**rigel_mu**: your raw model is untouched and remains the backbone; only the final string is canonicalized.

Current `rigel_test_all.csv` progress at 16:45: ~180k/485k. ETA ~17:25.

Tags: #postprocessing #ensembling #insight #model_selection #submission #status
