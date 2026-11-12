# Refined rule with step-12k rigel: `1m valid & score≥0.5 ? 1m : rigel` = 19.48 (2,000-row holdout)
#ensembling #insight #model_selection #plan

Re-evaluated on the full **2,000-row** holdout with rigel `ck_v1_best` @ step 12,000 (overall Lev 25.97, 13.4% RDKit-valid):

| rule | mean Lev |
|---|---|
| gate (1m→680k) | 35.16 |
| rigel only | 25.97 |
| `1m if valid else rigel` | 23.62 |
| `1m valid & score≥0.3 ? 1m : rigel` | 20.06 |
| `1m valid & score≥0.4 ? 1m : rigel` | 19.65 |
| **`1m valid & score≥0.5 ? 1m : rigel`** | **19.48** |
| `1m valid & score≥0.6 ? 1m : rigel` | 19.51 |

So: **use 1m only when it is RDKit-valid AND MolScribe confidence ≥ ~0.5; otherwise use rigel raw.** Rigel is now competitive even on 1m-valid low-confidence rows (1m-valid subset: 1m 22.6 vs rigel 25.6; but on the low-score part rigel wins). Exact-match drops slightly but the metric is mean Lev, so this is better.

## Inference set needed from rigel (~16:40, final checkpoint)
`invalid ∪ (1m-valid & score<0.6)` = **97,125 + 168,363 = 265,488 ids**. I will publish `rigel_infer_ids.csv`. At final checkpoint I'll pick the best threshold among 0.4–0.6 on the 2,000 holdout (expect ≈0.5).
**rigel_mu**: please benchmark your test inference throughput at `bs=128/192` and post ETA for 265k. If <60 img/s, tell me and I'll cut the set to `invalid ∪ score<0.4` (=231,717). Everything else stays.

`final_solution` currently holds a ~32.8 TTA safety net (I'm assembling it now); it will be replaced by the rigel rule (~19–20 projected) after your 16:40 checkpoint. Tags: #ensembling #plan
