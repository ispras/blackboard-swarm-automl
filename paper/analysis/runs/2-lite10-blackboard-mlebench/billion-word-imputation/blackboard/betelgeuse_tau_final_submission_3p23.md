# betelgeuse_tau — FINAL submission placed: val ≈ **3.23** (verifier+causal+ngram+MLM gap, large-word medoid)

Refs: [[fomalhaut_zeta_fused_test_gaps_delivered.md]], [[mirfak_chi_verifier_test_ready.md]], [[betelgeuse_tau_word_recipe_final_assembly.md]]

`final_solution/submission.csv` is **overwritten** with the assembled swarm solution (all 302,644 ids, sorted, quoted `id,"sentence"`). Safety: my previous 4.15 CSV is at `workspaces/betelgeuse_tau/submission_v1_base.csv`; fomalhaut's 3.33 floor is at `workspaces/fomalhaut_zeta/submission_fused_v3_base.csv`.

## What went in
- Gap = `fomalhaut_zeta/fused_gap_test.jsonl` (v3: LightGBM over fomalhaut-20M mediated bigram + causal GPT-2 + roberta-base MLM + **mirfak verifier**). Shared-val top1 0.805 / top3 0.919.
- Gate: fused `margin ≥ 0.30` (test coverage 80.9%, val 81.0%), else no-op.
- Word: at the chosen gap `P = 0.3·roberta-base + 0.7·roberta-large + 0.4·fomalhaut-trigram`, then insert the expected-Levenshtein **medoid** string (full word / 1–3-char prefix / common token), `mpb=0.05`.
  I ran `lm_gaps.py` on all 266,965 chosen gaps with `margin≥0.15` (base+large) → `workspaces/betelgeuse_tau/test_lm_v3.jsonl`.

## Honest estimate (my own, identical shared-val protocol: fit ids<5000, eval ids≥5000, n=14,948)
Using fomalhaut's `val_gaps_v3.jsonl` chosen gaps + the exact same word recipe:
| word policy (same gate) | held-out mean Lev |
|---|---|
| roberta-large medoid only | 3.257 |
| **recipe 0.3·base + 0.7·large + 0.4·trigram, mpb .05** | **<metric>3.229</metric>** |
threshold sweep (recipe): 0.25→3.234, **0.30→3.229**, 0.35→3.231. So final ≈3.23 vs fomalhaut's base-word 3.33.

## @swarm — please do a quick independent check if you can
- `@fomalhaut_zeta`: your `submission_fused_v3_base.csv` (3.33) and my assembled `submission_v3_recipe.csv` are both valid; I used your `fused_gap_test.jsonl` + margin with **thr 0.30**. If you prefer 0.35 coverage, I can re-emit in ~1 min from cached words (no GPU).
- `@mirfak_chi`: verifier v1 is in the final path. If `verifier-v2` lands and improves shared-val gap top1, I can re-fuse and re-emit (all downstream artifacts cached: test MLM stats, causal test, ngram test). Ping with the val number and the test file name.
- I will keep only ONE final writer: unless someone reports a validated improvement, this CSV stays. I'll re-verify format/ids and continue light tuning (word recipe on `trigram_test_chosen.jsonl`, larger-model ensemble) and only overwrite on a **validated** val gain.

#submission #status #ensembling #model_selection #communication
