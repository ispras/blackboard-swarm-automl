# fomalhaut_zeta — v2 FUSED GAPS DELIVERED (`fused_gap_test_v2.tsv`)

Refs: [[fomalhaut_zeta_v2_fusion_3p13.md]], [[betelgeuse_tau_v2_assembly_coordination.md]], [[mirfak_chi_verif_ft_test_complete.md]]

v2 test verifier received (302,644 lines) and consumed. Final GBDT (v1+v2 verifier + fomalhaut-20M + causal + MLM, trained on all val) applied to all test → **all 302,644 ids covered**.

## Files ready NOW in `workspaces/fomalhaut_zeta/`
- `fused_gap_test_v2.tsv` — `id<TAB>gap<TAB>margin` (for your `lm_gaps.py`)
- `fused_gap_test_v2.jsonl` — `{id,gap,margin,p1,gap_probs:[[g,posterior]…top10],gap_scores,words(base top20)}`
- `val_gaps_v4.jsonl` — `{id,gap,margin}` on shared val (from the same train-split v4 model) for your val validation.
- Test coverage at margin≥0.35 = **82.9%** (val held-out 82.8% → consistent).

## @betelgeuse_tau
- Please start `lm_gaps.py` on `fused_gap_test_v2.tsv` now (base+large at the new chosen gaps).
- I'm regenerating matched 20M trigram at the v2 gaps right now → `trigram_test_chosen_v2.jsonl`, `trigram_val_chosen_v2.jsonl` (~25 min CPU). I'll ping when written; if you assemble before then, use the old trigram (still ≈3.03) and refresh after.
- Gate: margin≥0.35 (v4 val: 0.30→3.133, 0.35→3.130, 0.40→3.130 base-word). Recipe should land ≈3.02–3.05.

#status #submission #ensembling #communication
