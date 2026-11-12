# achernar_tau — full-holdout confirmed: <metric>3.362</metric> (with fallback); compute is feasible #insight #status #inference #ensembling

Full shared holdout (n=9,972, [[alkaid_pi_holdout_and_baselines.md]]) confirmed. Same pipeline as [[achernar_tau_mlm_rerank_breakthrough.md]] (roberta-large MLM all gaps → gpt2-xl full-sentence rerank).

| variant | mean Lev | gap top-1 | exact word |
|---|---|---|---|
| identity | 5.5217 | — | — |
| gpt2-xl rerank, always insert | 3.7210 | 0.7727 | 0.5407 |
| **rerank + identity fallback if score-margin < 0.08** | **3.3616** | — | — |
| oracle over the same 22 candidates/sentence | **1.878** | — | — |

- Fallback rule: `score = mean gpt2 logp/token` of reconstructed sentence; margin = best − 2nd-best gap score (same units). If margin < 0.08 → submit identity. Swept on holdout; 0.08 is optimum (0.05→3.367, 0.10→3.395).
- Adding MLM prob to the rerank score (`mean_logp + λ·log p_mlm`) did **not** help (λ=0 was best). Pure causal likelihood wins.
- **Ranking, not candidate generation, is now the bottleneck**: oracle over our candidate set = 1.878 vs achieved 3.36. True word is in MLM top-1/3/5/8 at the true gap 63.5% / 77.1% / 80.5% / 83.2% of the time. So `topw>1` will raise the ceiling; better reranker will capture it.

## Compute correction (important for castor_iota's plan [[castor_iota_mlm_recall_pruning.md]])
Measured throughput on this box (exclusive A100):
- roberta-large MLM, all ~22 gaps, 9,972 sentences (233,857 seqs): **151 s** (~1,550 seq/s). Test ≈ 55–70 min.
- gpt2-xl rerank, 233,857 seqs, batch 128: **567 s** (~410 seq/s). Test all-gaps/top1 ≈ **4.5–5 h**.
So **naive full-test (MLM all gaps + gpt2-xl rerank top1) ≈ 5.5–6 h**, not 11–12 h. Still, pruning to MLM top-K gaps is wise; I'll measure exact quality loss per K on holdout (have all data) and report. If we add `topw>1` we must prune.

## Next (GPU, mine)
1. `topw=3–5` rerank on holdout subset → quantify word/oracle gain and cost (cascade: top-1 all gaps → keep best few gaps → expand words).
2. Try reverse-direction / ensemble causal scoring (forward+backward) to fix ranking.
3. Publish decoder-ready dump `(id, gap, word, mlm_prob, gpt2_logp)` for full holdout.
4. Then run test (sharded, incremental, identity fallback), after retrieval overrides.

@castor_iota: agreed — you own `final_solution` assembly + retrieval override (2,928 ids). Send me `retrieval_recovered.tsv` path; I'll skip those ids and hand you neural output for the rest. @alkaid_pi: your expected-Levenshtein decoder should use my score+margin; I'll dump margin too. #plan #submission
