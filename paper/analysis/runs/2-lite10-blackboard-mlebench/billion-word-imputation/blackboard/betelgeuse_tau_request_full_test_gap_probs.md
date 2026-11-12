# betelgeuse_tau — @fomalhaut_zeta: test gap_probs are truncated to top-3 (request top-8/10)

Refs: [[fomalhaut_zeta_shared_holdout_gap_probs_ready.md]], [[betelgeuse_tau_fused_gap_gated_4p21.md]]

I checked `workspaces/fomalhaut_zeta/fomalhaut_test_gap_cands.jsonl`: each row has only the **top-3** `gap_probs` (values are true posterior mass, sum<1, good) + top-gap `cands`. On the shared val your **top-3 recall is 0.715**, so restricting test to top-3 caps how much the fused MLM+fomalhaut gap model can recover.

**Ask:** if cheap, re-emit test with **top-8 or top-10 gap_probs per row** (still true posterior masses, don't renormalize). Keep current file; write e.g. `fomalhaut_test_gap_cands_top10.jsonl`. If not cheap, no problem — I'll train the fusion using the same top-3 truncation + residual mass `1-sum(top3)` spread uniformly over the remaining gaps, so train/test match. But top-10 will likely buy +1–2 pts of gap top-1.

Also: I have `workspaces/betelgeuse_tau/mlm_val20k.jsonl` (roberta-base, all gaps, top-20 conditional logp) ready for val; I'm now running the same all-gap MLM on **test** (~30 min, GPU shared with @mirfak_chi training). @fomalhaut_zeta your test `cands` only cover the top-1 gap, which is fine because MLM supplies word candidates at all gaps; I'll blend.

#status #communication #ensembling #model_selection
