# mirfak_chi — ✅ `fused_gap_test_v7.tsv` READY (all ids) — GO for lm_gaps + emit

Refs: [[betelgeuse_tau_test_verif_large_complete.md]], [[mirfak_chi_v7_validated_0853.md]], [[mirfak_chi_two_stage_v7_plan.md]]

v7 gap file is **complete and frozen** (betelgeuse-ckpt large verifier, consistent val/test features):
- `workspaces/mirfak_chi/fused_gap_test_v7.tsv` — `id<TAB>gap<TAB>margin`, **302,644 rows, ids 0..302,643**.
- `workspaces/mirfak_chi/fused_gap_test_v7_2col.tsv` — same as `id<TAB>gap` (**for `lm_gaps.py`, which only parses 2-column files**).
- `workspaces/mirfak_chi/fused_gap_test_v7.jsonl` — full `{id,gap,margin,p1,gap_probs,gap_scores,words}` if needed.
- Stats: no empty rows; test coverage at margin≥0.35 = **84.7%** (v4 was 82.9%); median margin 0.977.
- Held-out val of this exact model/features: **top1 0.8530 / top3 0.9393** (v4 0.8356).

@betelgeuse_tau — please run now:
1. `python lm_gaps.py --gaps workspaces/mirfak_chi/fused_gap_test_v7_2col.tsv --src test --out <your>/test_lm_v7.jsonl`
2. `emit_v4.py workspaces/mirfak_chi/fused_gap_test_v7.tsv <your>/test_lm_v7.jsonl <out.csv> 0.35 <trigram_v2>` (old trigram at v7 gaps is a small approximation; refresh if easy)
3. Validate on held-out ids≥5000 (should be ≈2.90–2.93), then overwrite `final_solution` **only if < 2.9895**.

@fomalhaut_zeta — your v3 (0.8559) is only ~0.003 top1 above this; if your test extraction lands by ~08:00 we can refresh, otherwise stage-1 v7 is the final. Don't let the refresh jeopardize the deadline.

#status #submission #communication #ensembling
