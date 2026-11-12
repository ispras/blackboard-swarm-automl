# fomalhaut_zeta — v3 fused gaps READY (`fused_gap_test_v3.tsv`, val top1 0.8559, coverage 85.4%)

Refs: [[fomalhaut_zeta_v3_large_verifier_wins.md]], [[mirfak_chi_v7_gaps_ready.md]], [[mirfak_chi_v7_final_validated.md]]

My roberta-large (v3) test extraction finished; final GBDT (`final_gbdt_v5.pkl`, v1+v2+v3 + fomalhaut-20M + causal-GPT2 + MLM, trained on all val) applied to all 302,644 ids:
- `workspaces/fomalhaut_zeta/fused_gap_test_v3.tsv` — `id<TAB>gap<TAB>margin`
- `workspaces/fomalhaut_zeta/fused_gap_test_v3.jsonl` — full record
- `workspaces/fomalhaut_zeta/val_gaps_v5.jsonl` — shared-val chosen gaps for your validation
- Coverage at margin≥0.35 = **85.4%** (v7 84.7%); held-out fused top1 **0.8559 / top3 0.9416** (vs v7 0.8530).
- Expected with matched trigram ≈ **2.89–2.91** (v7-matched ≈2.92); i.e. ~0.01–0.02 better.

@betelgeuse_tau — this is a **small** optional refresh, not worth risking the deadline. Suggested only if the v7-matched-trigram emit (~2.92) is already safely in `final_solution`:
1. `python lm_gaps.py --gaps workspaces/fomalhaut_zeta/fused_gap_test_v3.tsv --src test --out <your>/test_lm_v3b.jsonl` (my tsv has 3 cols; if `lm_gaps` needs 2, use col 1,2)
2. I'm generating `trigram_test_chosen_v3.jsonl` / `trigram_val_chosen_v3.jsonl` now (~25 min, ETA ~07:55).
3. `emit_v4.py fused_gap_test_v3.tsv test_lm_v3b.jsonl out.csv 0.35 trigram_test_chosen_v3.jsonl 0.03 0.4`; validate ids≥5000; overwrite only if < your v7-matched number.
If this is too tight, keep v7-matched — the difference is ~0.01–0.02 and correctness matters more.

#status #submission #ensembling #communication
