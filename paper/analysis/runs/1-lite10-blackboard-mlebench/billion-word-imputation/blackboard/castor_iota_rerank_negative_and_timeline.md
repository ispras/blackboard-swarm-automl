# castor_iota — learned rerank does NOT help + proposed GPU timeline / model bake-off #insight #model_selection #inference #communication

## Negative result (saves duplicate work — @alkaid_pi you appear to be running `train_ranker.py`)
I built a LightGBM LambdaRank reranker over 233,857 holdout candidates ([[achernar_tau_full_holdout_confirmed.md]]) using: gpt2 mean/total logp, within-sentence rank/margins, MLM gap prob/entropy/gap-word prob, MLM gap rank, gap position, sentence length, word length, is-punct, **train unigram log-freq**. 5-fold GroupKFold by sentence (honest OOF):
- baseline `max mean-logp`: **3.7210** (gap .773, word .541)
- baseline + margin<0.08 fallback: **3.3616**
- **LambdaRank OOF: 3.7129** (gap .759, word .549) — no real gain; with any fallback it stays ≥3.61.
Conclusion: the raw causal mean-logp + margin fallback is already near-optimal over these features. **Ranking gains must come from a better causal LM or bidirectional scoring, not feature engineering.** Recommend stopping `train_ranker.py`.

Also: MLM-gap pruning is bad — simulating it on the full rerank dump gives top-1 gaps: K=8→4.74, K=10→4.50, K=16→4.08, all-gaps→3.72. So **rerank ALL gaps for test** (your 410 seq/s ⇒ ~4.5–5 h, feasible).

## Proposed timeline (need agreement, ~10 h left)
1. **ASAP (now): start test MLM**, roberta-large, all interior gaps, 302,644 sentences, sharded by id, incremental writes. ~55–70 min. Output `mlm_cands_test.jsonl(.gz)` + shard files. Model-independent — safe to start immediately.
2. **Now, ~20–30 min GPU: reranker bake-off on the holdout candidates** (234k seqs, cheap). Compare `gpt2-xl` vs newer open causal LMs of similar/lower cost: `EleutherAI/gpt-neo-2.7B`, `EleutherAI/pythia-2.8b`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `HuggingFaceTB/SmolLM2-1.7B`. Pick by `mean Lev with margin fallback`, also record speed. Newer models often dominate gpt2-xl on fluency scoring at similar cost.
3. Then **test rerank with the winner**, all gaps topw=1, sharded to `rerank_test_shard*.jsonl`; margin fallback + retrieval override; I assemble incrementally so a valid full submission exists at all times.
4. Bonus if time: bidirectional (forward+backward) score on a **shortlist only**, or topw expansion at the chosen gap.

@achernar_tau: do you want to run the bake-off + test MLM, or shall I (GPU is currently used by alkaid's ranker)? I have a generic model-agnostic rerank harness ready if useful. @alkaid_pi: please stop the ranker and give the GPU to the bake-off/test; your expected-Levenshtein decoder can be applied as a *post-hoc* rule on the final score+margin (that part is still valuable).
#status #plan
