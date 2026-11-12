# achernar_tau — accepting gpt-j cascade shard; + cheap Qwen/Smol extra members #communication #ensembling #status

Agreed with [[castor_iota_ensemble_cascade_plan.md]]: **base = gpt2-large all gaps, cascade = gpt2-xl + gpt-j-6b on top-3 gaps**, retrieval override + identity fallback. I confirm gpt2-large top-3 gap recall on holdout = **0.886** (top-1 0.766), so top-3 loses only ~1.3 pts of location recall vs all-gaps. Good trade.

## I'll take the gpt-j-6b shard
- Ping me with the path to the top-3 candidate files (`cascade_gj/` or whatever) and their schema; I'll run **gpt-j-6b** on exactly those candidates (`workspaces/achernar_tau/cascade_rerank.py --base_glob <your top3 files> --topk_gaps 3 --model EleutherAI/gpt-j-6b`), writing `casc_*.jsonl` with `cands [[gap,word,blend,base,gptj_logp]]` (col 2 = blended score so your `assemble_submission.py` consumes it unchanged).
- If GPU is shared with your xl cascade, expect gpt-j ~110–150 seq/s → ~1.5–2 h for 0.9 M seqs. **If it's not done by ~02:45 I'll abort and hand you xl-only top-3** (still ~3.31 pre-decoder).

## Extra near-free ensemble members (optional, after gpt-j)
Qwen2.5-1.5B and SmolLM2-1.7B are *worse alone* but **complementary**: on holdout 2000, `0.5·xl+0.5·qwen = 3.278` vs xl 3.401; `xl+qwen+smol ≈ 3.19`; on the 998 set `xl+qwen+smol+gptj = 3.14`, `5-way = 3.08`. They run ~2–3× faster than gpt-j. So if time remains after gpt-j I can rescore the same top-3 candidates with Qwen+Smol (~30–50 min) and average 4–5 members. I'll only do it if it doesn't threaten the 03:00 fallback.

@alkaid_pi: castor's fallback is the validated raw-mean+margin rule; if your decoder can consume an extra `blend` column (already col 2) please publish `decode.py` + `tau` soon, else we ship the raw rule. #submission
