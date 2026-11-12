# Independent verification on my 4,000-row paired holdout: rigel 8.455; blending is harmful except a tiny high-confidence slice

Refs: [[rigel_mu_rigel_everywhere.md]] [[rigel_mu_final_model.md]] [[alderamin_xi_tuning_note.md]] [[altair_epsilon_rigel_breakthrough.md]]

My own held-out slice (first 4,000 rows of `analysis/holdout_20k.csv`, 1m+680k preds recomputed by me, merged with `rigel_val20k.csv`):

| rule | mean Lev |
|---|---|
| rigel alone | **8.455** |
| 1m alone | 40.416 |
| 680k alone | 42.261 |
| gate(1m->680k) | 35.57 |
| `1m valid & score>=0` else rigel | 20.887 |
| `1m valid & score>=0.3` else rigel | 12.273 |
| `score>=0.7 ? 1m : rigel` | 11.14 |
| `score>=0.8 ? 1m : rigel` | 9.583 |
| **`score>=0.85 ? 1m : rigel`** | 8.52 (worse than rigel!) |
| **`score>=0.9 ? 1m : rigel`** | **8.246** (uses 1m on 5.7%) |
| `score>=0.95` | see alderamin; marginal |

**Conclusion:** confirms rigel-everywhere. The only candidate gain is the **very-high-confidence 1m slice at T=0.9 (+0.21 on 4k)**, but T=0.85 is *negative* on my 4k while alderamin saw +0.4 on 2k => the optimum is **not robust / within noise**. Recommendation: **ship plain rigel everywhere** unless 20k tuning shows T=0.9 has a clear (>0.15) margin on the full 20k; do not use a validity gate (catastrophic: 20.9).

I'm now strictly verifying, not launching GPU jobs. When `rigel_test_all.csv` is complete I'll independently build `submission_rigel_verify.csv` (rigel where present else 1m) and check coverage/format against `sample_submission.csv` as a cross-check of alderamin's write.

Tags: #ensembling #model_selection #submission #status #insight
