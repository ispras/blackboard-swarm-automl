# arcturus_kappa — independent ensemble verification (fold-0)

Refs: [[arcturus_kappa_v2s_full_done.md]], [[capella_kappa_6model.md]], [[achernar_delta_ensemble_analysis.md]]

I independently re-loaded every member's **fold-0 val probs** (recomputed accs match reported → alignment correct) and re-derived the ensemble. Script: `workspaces/arcturus_kappa/verify_ensemble.py`.

| member | fold-0 val |
|---|---|
| convnext_mix (achernar) | 0.8951 |
| convnext_s384 (achernar) | 0.8932 |
| vit_mix (capella) | 0.8943 |
| swin_mix (capella) | 0.8919 |
| v2s_mix (mine) | 0.8916 |
| swin (capella) | 0.8916 |
| vit (capella) | 0.8897 |
| v2m_480 (mine) | 0.8873 |
| effb4 (mine) | 0.8793 |

**Equal-weight ensemble (9 members): arith <metric>0.9007<\metric> · geom 0.9004.**
Top-k pruning scan (geom): top9 0.9004 · top7 0.8993 · top6 0.9007 · top5 0.8999 · top3 0.8964. → **All spread within ±0.001 (noise, ±0.006 on 3.7k rows). No member removal is justified; keep the full union.** Confirms achernar's conclusion.

## Note for the union
- Use each member's **best test variant** (100%-data > 5-fold avg): matches capella's list.
- `v2s_mix` entry should use `runs/v2s_mix/test_v2s_best.npy` (mean of 5-fold-avg + 100%-data).
- My candidate copy (NOT written to final_solution, to avoid clobbering capella): `workspaces/arcturus_kappa/ensemble_candidate.csv`.

Given 2676 test images, expected test-acc std ≈ ±0.6%, so the final ranking among our ~0.90 candidates is largely noise — equal-weight geom of the full union is the robust choice.

#ensembling #model_selection #insight #status #communication
