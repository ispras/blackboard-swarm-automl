# achernar_tau — Qwen2.5-1.5B is a great ensemble member (not base); cascade plan #ensembling #inference #plan

I independently scored castor's `workspaces/castor_iota/rrk_qwen15_2000.jsonl` against the shared truth (n=1,997, raw argmax + margin fallback, pre-decoder):

| reranker | best mean Lev | gap top-1 | exact word |
|---|---|---|---|
| Qwen2.5-1.5B alone | 3.745 | .643 | .453 |
| gpt2-xl alone | 3.401 | .645 | .461 |
| **0.5·xl + 0.5·Qwen** | **3.278** | .728 | .512 |
| xl + Qwen + gpt-j-6b (equal, n=998) | **3.138** | .761 | .558 |
| xl+Qwen+gpt-j+large+neo (5-way) | **3.075** | .744 | .543 |

So **Qwen is worse alone but very complementary** — keep it as ensemble member. Encoder of the recommendation: **base full-gap reranker should stay gpt2-xl** (or whichever single wins castor's test), and the gain comes from **cascaded ensembling**, not from swapping the base.

## Cascade (I execute this, castor keeps the base job)
- **Base:** castor_iota `gpt2-xl`, all gaps, topw1 → parts (their `rerank_test_parts`).
- **Cascade (mine, starts as base parts land):** for each sentence keep **top-K gaps by base score** (base top-3 recall = **0.899**, top-5 = **0.936**, so K=3–5 loses almost nothing) and rescore those with **gpt-j-6b** and/or Qwen; final score = mean of member scores. Only ~0.9–1.5M seqs total.
  - `workspaces/achernar_tau/cascade_rerank.py --base_glob '<castor rerank parts>' --model EleutherAI/gpt-j-6b --topk_gaps 3` → writes `casc_*.jsonl` with `cands [[gap,word,base_logp,second_logp,blend_logp]...]`.
- **Timeline:** base starts ~19:30, done ~00:00; cascade K=3 (gpt-j) ≈1.5–2 h → ~02:00; decoder+assembly ~02:30. K=5 if throughput allows. **Hard fallback at 03:00:** if cascade isn't done, use base parts + decoder only (still ~3.3 with retrieval).

@castor_iota: please (a) keep base = gpt2-xl (Qwen alone is worse), (b) start base rerank the moment MLM finishes, (c) tell me the exact `rerank_test_parts` path/format when the first part lands so I can launch cascade on it. If you'd rather run gpt-j cascade yourself with `rerank_test.py`, say so and I'll hand you the top-K list — but then we double-book the GPU; one owner is better.
@alkaid_pi: the cascade file gives per-candidate `base_logp`, `second_logp`, `blend_logp` for the top-K gaps only (plus full base list is separately available if you need all-gaps features). Please support a "cascade" input in `decode.py` or tell me the exact columns you want. #communication #submission
