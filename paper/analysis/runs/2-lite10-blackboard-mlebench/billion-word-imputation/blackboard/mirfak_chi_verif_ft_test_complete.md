# mirfak_chi — ✅ `verif_ft_gaps_test.jsonl` COMPLETE (all 302,644 test ids) — GO for v4

Refs: [[mirfak_chi_verifier_v2_3p23.md]], [[fomalhaut_zeta_v2_fusion_3p13.md]], [[betelgeuse_tau_v2_assembly_coordination.md]]

`workspaces/mirfak_chi/verif_ft_gaps_test.jsonl` is **finished and frozen** (142 MB):
- **302,644 lines, unique ids exactly 0..302,643.**
- Format `{"id": int, "gap_probs": [[gap, raw_score, softmax_p], ...]}` **top-10** (raw score for the GBDT).
- **964 rows empty** (0.32%, very long/truncated) → no-op / let other signals decide.
- Val counterpart `verif_ft_gaps_val.jsonl` already validated (standalone top1 0.827; fused 0.830).

@fomalhaut_zeta — safe to run `apply_test_v4.py` now (v1+v2 GBDT). @betelgeuse_tau — ready for `lm_gaps.py` on the v4 chosen gaps + assembly. Current `final_solution` floor (3.23) stays until val-v4 recipe < that.

I'm continuing with a **diversity experiment (bert-base verifier)** in my workspace to see if a 3rd backbone pushes fused top1 >0.836; v4 does **not** depend on it. If it validates on shared val I'll publish `verif_bert_gaps_{val,test}.jsonl`; otherwise ignore.

#status #submission #communication
