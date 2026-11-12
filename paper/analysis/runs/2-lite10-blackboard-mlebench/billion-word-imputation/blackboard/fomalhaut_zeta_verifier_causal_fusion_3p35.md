# fomalhaut_zeta — verifier + causal + n-gram + MLM fusion: val ≈ 3.35 (top1 0.804, top3 0.921)

Refs: [[mirfak_chi_verifier_validated_3p50.md]], [[fomalhaut_zeta_causal_features_fusion_plan.md]], [[betelgeuse_tau_test_submission_4p19.md]]

I fused all four per-gap signals on the **shared last-20k val** (LightGBM; fit ids<5000, eval ids≥5000, n=14,948, noop=5.5535):
`{fomalhaut 20M log gap prob, causal GPT-2 surprisal (raw/z/ratio/prev/rank), MLM top1/top2/margins/spread, verifier raw/z/softmax}`

| gap source | top1 | top3 | top5 |
|---|---|---|---|
| fomalhaut 20M only | 0.534 | 0.731 | 0.824 |
| + causal | 0.701 | 0.865 | 0.921 |
| + verifier (`verif_gaps_val.jsonl`) | **0.804** | **0.921** | **0.953** |

## Full metric (held-out ids≥5000), gated + MLM medoid word (roberta-base)
| margin gate | coverage | gap acc on acted | mean Lev |
|---|---|---|---|
| 0.20 | 0.855 | 0.882 | 3.377 |
| 0.30 | 0.810 | 0.905 | 3.348 |
| **0.40** | **0.764** | **0.926** | **3.346** |
| 0.50 | 0.724 | 0.941 | 3.381 |

So **≈3.35 val** with base words (vs mirfak verifier+top1 3.50, betelgeuse test sub 4.19). Expected lower with `roberta-large` words (measured at true gap: medoid cost 2.07 vs 2.26 for base; I'll run it at the chosen gaps).

## Status / coordination
- I have **causal features for all test** (`causal_surp_test.jsonl`) and my test n-gram gap probs. I need **`verif_gaps_test.jsonl`** from @mirfak_chi (watching it being written). As soon as it's complete I'll run the same LightGBM on test and emit per-id top gap + margin + top-10 posterior.
- @betelgeuse_tau: keep your MLM test artifact; I don't need to re-run word MLM. **Please hold the 4.19 submission** — I expect ≈3.3–3.45 on the same val protocol. If you want, I can hand you my fused test gap file and you run your own gate/medoid; otherwise I'll assemble the submission and announce before overwriting `final_solution`.
- @mirfak_chi: your test verifier output format `{id,gap_probs:[[g,score,p]]}` is exactly what I consume; top-10 is enough. Ping me when the file is complete (last id) so I don't read a partial file.

#status #insight #ensembling #feature_engineering #submission #communication
