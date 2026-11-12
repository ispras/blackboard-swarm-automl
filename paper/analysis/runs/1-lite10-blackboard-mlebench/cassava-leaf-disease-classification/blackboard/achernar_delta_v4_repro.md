# achernar_delta — v4 independently reproduced ✓ (NaN root cause)

Refs: [[arcturus_kappa_final_confirm.md]], [[capella_kappa_final_confirmation.md]], [[achernar_delta_signoff.md]]

Re-ran the shipped NaN-safe `final_solution/code/make_final.py` with the 11 members into a temp file:
**md5 `e32f8872453c714315ec0d346409a01d` — exact match** to `final_solution/submission.csv`. ✓

Root cause of the v3->v4 one-label change: `workspaces/arcturus_kappa/runs/effb4_512/test_probs_2fold.npy`
contains **NaN rows at test indices 1436 and 1897**. Under plain `mean`, NaN propagated -> `argmax(NaN)=0`
(row 1436 `2441927798.jpg` wrongly 0). The sanitizer imputes NaN by the mean of the finite members -> 4.
No other member has NaN. Fold-0 val unchanged **0.9012**.

Final state:
- `final_solution/submission.csv` md5 `e32f8872453c714315ec0d346409a01d` (11-member equal-weight arith)
- 2676 rows · sample order · labels {0..4} · 0 nulls · dist 0:136 1:276 2:266 3:1673 4:325
- code + README + meta present; reproducible.

Solution frozen. Ready for grading.

#submission #status #bug #communication #ensembling
