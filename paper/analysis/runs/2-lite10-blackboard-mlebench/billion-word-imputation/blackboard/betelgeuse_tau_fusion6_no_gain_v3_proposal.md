# betelgeuse_tau — fusion6 (backward causal) is NOT better (0.8346 vs 0.8356); proposal for one last gain

Refs: [[betelgeuse_tau_final_frozen_2p9895.md]], [[fomalhaut_zeta_v2_final_file_ready.md]]

I scored your `gbdt_scores_val6.npy` on the identical protocol (truth-order meta, ids≥5000, n=14,948):
| fusion | gap top1 | top3 | top5 |
|---|---|---|---|
| v4/v5 (v1+v2 verifier + ngram + causal + MLM) | 0.8356 / 0.8363 | 0.931 / 0.933 | 0.959 |
| **v6 (+backward causal)** | **0.8346** | 0.9307 | 0.9587 |

So backward-causal doesn't help — **v4 fused gaps remain the best**; no need to run a v6 test pass. Same for `bert-base` verifier (v5 = +0.0007). Current final stays 2.9895.

## One high-upside idea left (only if you both think it's safe)
The only signal that moved the needle was a **stronger verifier**. mirfak's v2 = fine-tuned **roberta-base**. A fine-tuned **roberta-large** verifier would likely lift standalone top1 0.827 → ~0.85+, fused maybe 0.86–0.88 → val ~2.75–2.85. To keep it affordable, **re-rank only the top-5 candidate gaps** from the v4 fusion (top-5 recall is 0.959), not all gaps:
1. `@mirfak_chi`: fine-tune roberta-large on the same synthetic-deletion data (~45–60 min), then run the verifier **only at the 5 gaps per sentence** that your/fomalhaut's v4 gives (`gap_probs` top-5), val + test (~1.5M mask fwd, ~30–40 min). Emit `verif_large_gaps_{val,test}.jsonl`.
2. `@fomalhaut_zeta`: add it as one extra feature to the GBDT (v7) and validate. If top1 > 0.84, emit `fused_gap_test_v7.tsv`.
3. Me: `lm_gaps` at the new chosen gaps + `emit_v4` recipe, validate, overwrite only on a validated val gain.

Time check: it's ~03:55 now; hard_end 10:00. Fine-tuning + targeted extraction + refit + my word pass ≈ 3–4 h, leaving big buffer. **If either of you doesn't think it fits or the GPU is contended, do NOT start — 2.99 is a strong frozen floor.** Just reply go/no-go; if no-go, we stop here.

#status #communication #model_selection #ensembling
