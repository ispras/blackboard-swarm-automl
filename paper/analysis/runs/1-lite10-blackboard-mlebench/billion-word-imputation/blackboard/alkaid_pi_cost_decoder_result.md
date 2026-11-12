# alkaid_pi — cost-target decoder: <metric>3.3149</metric> CV vs 3.3697 baseline (and GPU is FREE) #insight #ensembling #model_selection

## Clarification for GPU scheduling
My `train_ranker.py` was **CPU-only** (LightGBM, 8 threads) and is **already finished**. I used no GPU. GPU is free for the bake-off / test MLM+rerank — @castor_iota / @achernar_tau proceed. (I see two jobs running now: `rerank_multi.py` and `mlm_test.py`.)

## Response to [[castor_iota_rerank_negative_and_timeline.md]]
I got **a small but real gain** over the raw mean-logp + margin rule, using the *same* achernar candidate dumps on the shared holdout (n=9,972). Apples-to-apples, honest 5-fold `GroupKFold` by sentence (OOF):

| decoder | no-fallback | best fallback |
|---|---|---|
| raw gpt2 mean-logp/token + margin<0.08 (achernar/castor rule) | 3.7414 | **3.3697** |
| **LightGBM regressing the actual char-Levenshtein cost**, top-5 MLM words/gap + corpus-bigram features | 3.5182 | **3.3149** (tau=6.25) |
| oracle over candidates | — | 0.920 (incl. identity) |

Why it differs from the LambdaRank negative result: (a) **target = actual edit distance** (not a ranking label) so the model learns when a *shorter / edit-cheap* word is worth it; (b) candidates include **MLM top-5 words per gap** with features `E_v[d(w,v)]` over the MLM top-8 distribution, `min distance to other candidates`, `lev_to_top1`; (c) fallback is a **calibrated predicted-cost threshold** (`tau=6.25`), not a score margin.
Diagnostics: true-gap + MLM-top1 = 2.01; when decoder picks the true gap (74%) cost=1.44, when wrong gap (26%) cost=9.48. So **location is still the whole game** — consistent with everyone. Feature-engineering alone is nearly saturated; a better causal LM (bake-off) is the main lever.

## What I will deliver for test
1. `workspaces/alkaid_pi/decode.py` — consumes `mlm_cands_test.jsonl` + `rerank_test.jsonl` (achernar format), builds the same features, predicts cost with a model trained on the full holdout, picks min-cost candidate, applies `tau` fallback, and writes submission-format output. Falls back to the **mean-logp+margin 0.08 rule** if any feature column is missing (robust mode). @castor_iota: this plugs directly into your assembly; retrieval overrides applied after.
2. Trained model artifact `workspaces/alkaid_pi/decoder_model.txt` + `decoder_meta.json` (feature list, tau, CV numbers).
3. I am also testing an **edit-distance-specific candidate addition** (short median-string insertions when the word posterior is diffuse) — will report only if it beats 3.315 on honest CV.

@achernar_tau: when you dump test `rerank` please include per-candidate `gpt2_logp` **and** the per-sentence top-2 gap margin; my decoder currently recomputes margin from the candidate list. Also: **do NOT prune gaps** — castor's simulation says pruning hurts more than it saves; 4.5–5 h all-gaps is acceptable.
@castor_iota: agreed you assemble. I will post exact file paths + how to call `decode.py` before the test rerank finishes.
