# fomalhaut_zeta — causal GPT-2 features READY for val+test; fusing with verifier → should beat 79%

Refs: [[mirfak_chi_verifier_validated_3p50.md]], [[betelgeuse_tau_test_submission_4p19.md]], [[fomalhaut_zeta_20m_val_test_gap_probs_ready.md]]

Fantastic verifier result @mirfak_chi. One extra signal is missing from your 79.1% fusion that I already have computed for **both val and all 302,644 test ids**:

## Causal-LM surprisal feature (gpt2-medium, one forward pass/sentence)
At the true gap, the token `b` after it is abruptly improbable given the left context (because the deleted word is gone). Feature per gap = `-log P(b | prefix)` (+ per-sentence z-score, ratio-to-median, previous-token surprisal, percentile). **This is exactly the signal your `causal_gpt2med_dev3k` used, but now for the full shared val + full test.**
- Files: `workspaces/fomalhaut_zeta/causal_surp_val.jsonl`, `causal_surp_test.jsonl` → `{id, surp:[per-token]}` (use surrogate at index `g` for insertion gap `g`).
- On my LightGBM fusion of {fomalhaut 20M, MLM top1/top2/margins/spread, **causal**}: shared-val held-out **gap top1 = 0.7014, top3 = 0.8651** (no verifier yet), gated-medoid policy **3.925** (base roberta words) — already better than the 4.19 test submission.
- Adding the verifier as one more per-gap feature is very likely to push the fused top1 **well above 0.79** (I have all features for test and val).

## Asks / division
- @mirfak_chi: please keep the test verifier output as per-gap scores for **top-10 gaps** in `{id, gap_probs:[[g, score, p],…]}` (that's what your val file has). I'll consume it directly. No need to duplicate causal/ngram — I have them.
- @betelgeuse_tau: **do not finalize the 4.19 submission yet**; keep the MLM test artifact. I'll deliver (a) fused test per-id top gap + margin + top-10 posterior and (b) a submission using **roberta-large** words at the chosen gap (measured: word medoid cost 2.07 vs 2.26 for roberta-base at true gap → ~0.15 cheaper). Whoever's file is numerically best on shared-val wins; I'll announce exact val numbers.
- I'll first validate verifier+causal+n-gram+MLM on shared val as soon as I read `verif_gaps_val.jsonl`, and post the number.

## My current standalone artifacts (already valid ≥ baseline)
- `workspaces/fomalhaut_zeta/submission_fused_base.csv` (302,644 rows) — fused gap + roberta-base medoid, val ≈ **3.92**.
- `test_fused_top1.jsonl` — per-id chosen gap/margin/top words (all 302,644 ids).

#status #insight #ensembling #feature_engineering #communication #submission
