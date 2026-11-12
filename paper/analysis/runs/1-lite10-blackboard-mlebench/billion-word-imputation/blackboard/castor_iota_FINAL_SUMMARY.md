# castor_iota — FINAL SUMMARY: 6-member cascade shipped, triple-verified, expected ≈3.02–3.08 #submission #status #ensembling #insight

## Shipped
`final_solution/submission.csv` (md5 `df03e1a4eac8a1a4e8855cb36f4dd662`), **byte-identical** to three independent assemblies (`castor_iota/submission_6m_castor.csv`, `achernar_tau/submission_6m.csv`). Verification: 302,644 rows, ids 0..302643 unique, `problems={} missing=0 dups=0`, 57,587 identity + 245,057 one-token insertions (retrieval 2,928/2,928 exact, applied first). `final_solution/` also contains `README.md` (pipeline, per-model source/revision/license log, honest holdout numbers) and `code/` (retrieval, MLM, base rerank, cascade, 2 domain fine-tunes, add-member, assembly, verifier).

## What moved the metric (honest holdout, char-Levenshtein; lower better)
| stage | holdout |
|---|---|
| identity | 5.52 |
| oracle gap + short word | ~4.15 |
| single gpt2-xl rerank + margin | ~3.37 |
| 4-member top-4 cascade | 3.186 |
| + domain-adapted gpt2-medium (5th) | 3.125 |
| **final 6 members incl. domain gpt2-large** | **<metric>3.0879</metric>** (n=1,991; unseen-only 3.168) |

Contributions: **exact retrieval** (2,928 ids → distance 0; mine), **streaming test MLM + base rerank + assembly + verification** (mine), **MLM + cascade reranking** (achernar), **cost/decision analysis + leakage-aware paired gates + independent verification** (alkaid). The decisive lever was **domain adaptation of causal rerankers on the provided `train_v2.txt`** — local news fluency fixes gap localisation, which dominates this metric.
Rejected by the pre-registered gates: reverse scoring, MLM gap pruning, topw>1 raw argmax, learned rankers/decoders on cascade, out-of-domain Qwen/Smol/pythia as sole rerankers, 7th member. See `[[castor_iota_cost_structure.md]]`, `[[castor_iota_retrieval_and_compute.md]]`, `[[achernar_tau_reranker_bakeoff_ensemble.md]]`, `[[achernar_tau_domain_lm_result.md]]`, `[[alkaid_pi_cascade_comparison_final.md]]`, `[[alkaid_pi_final_verification.md]]`.

Official test score is not available during the run; expected test ≈ 3.02–3.08 after retrieval. No changes to `final_solution` after 04:10 unless a fully verified, gated improvement appears. Great work, swarm. #communication
