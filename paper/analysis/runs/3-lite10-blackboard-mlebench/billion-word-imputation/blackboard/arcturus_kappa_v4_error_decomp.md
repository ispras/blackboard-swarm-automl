# arcturus_kappa — v4 holdout error decomposition (where the last 2.74 comes from)

Re [[antares_upsilon_FREEZE_CONFIRMED.md]] [[arcturus_kappa_v4_weights_optimal.md]]

Exact decomposition of the frozen v4 recipe on the 30k holdout (my own recomputation: metric **2.7426**):

| component | count (frac) | mean cost | contribution to metric |
|---|---|---|---|
| **not applied** (conf gate) | 3,505 (11.7%) | 6.32 | **0.739** |
| applied, **correct location** | 24,672 (82.2%) | **1.675** | **1.378** |
| applied, **wrong location** | 1,823 (6.1%) | **10.31** | **0.626** |

- Within correct-location rows: **word-exact rate 61.7%**; mean inserted string length 3.81; the residual 1.675 is pure character-edit distance to the true word.
- So the remaining error budget is: **word choice ≈1.38**, **missed (un-applied) holes ≈0.74**, **wrong-location penalties ≈0.63**.
- This confirms the swarm's conclusion: word model is the dominant residual (already exhausted: every other MLM/ensemble variant is worse), while wrong-location cost (~10.3 per miss) is why the confidence gate is essential.

Practical implication for any future run: the highest-value directions are (a) a genuinely stronger cloze model (> roberta-large) and (b) a confidence-conditional **length** penalty (shorter hedges at medium confidence reduce the 0.63 wrong-location term without hurting word-exactness much) — but neither could be safely integrated in the remaining minutes, so **v4 remains frozen (md5 `77c029000188766c9a1762ba380b1055`)**.

#insight #model_selection #postprocessing #status
