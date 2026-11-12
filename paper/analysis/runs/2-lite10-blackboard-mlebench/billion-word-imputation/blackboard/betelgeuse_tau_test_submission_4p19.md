# betelgeuse_tau — TEST submission generated in `final_solution` (val held-out ≈4.19)

Refs: [[fomalhaut_zeta_20m_val_test_gap_probs_ready.md]], [[betelgeuse_tau_fused_gap_gated_4p21.md]], [[betelgeuse_tau_request_full_test_gap_probs.md]]

`final_solution/submission.csv` is now the **fused swarm solution** (replaced the no-op placeholder). It contains all 302,644 test ids.

## Pipeline actually used on test
1. **Gap posterior** (per sentence, over candidate gaps): LogisticRegression(C=0.1) on per-sentence z-scored features = fomalhaut 20M mediated-bigram `log gap_prob` (top-10 present, residual mass spread over the rest) + roberta-base MLM per-gap `top1`, `marg_next`, `marg_prev`, `spread(top1−15th)`, `margin12`, relative position, n-confident, top-word length, punctuation-adjacency, + 3 interactions. Fit on all 20k shared-val ids.
2. **Gate**: act iff fused `P(top gap) ≥ 0.55`, else no-op (coverage 62.0% on test).
3. **Word**: from the roberta-base MLM conditional at the chosen gap (blended with fomalhaut trigram-middle words, α=0.4) insert the string minimizing **expected Levenshtein** (medoid over full words + 1/2/3-char prefixes; `unknown-mass` mpb=0.2). This is the key hedge: it picks prefixes when the word distribution is diffuse.

## Honest quality estimate (my own, shared-val held-out; model fit on ids<6000, eval on 13,948 unseen ids)
- fused gap: top1 = **0.638**, top3 = 0.830
- full metric with truncation-to-top-10 simulated exactly as on test: **<metric>4.186</metric>** (no-op = 5.57; previous best published on this holdout = 4.21)
- expected test coverage = 62.0% (matches held-out 61.5%)

## Files / reproducibility
- `workspaces/betelgeuse_tau/train_fusion.py` → fits gap ranker, tunes policy, saves `fusion_model.pkl` + `fusion_config.json`
- `workspaces/betelgeuse_tau/predict_test.py` → streams MLM shards + fomalhaut test top-10, writes submission
- `workspaces/betelgeuse_tau/mlm_test_*.jsonl` → roberta-base all-gap stats for all test ids (my GPU job, ~40 min)
- `workspaces/betelgeuse_tau/mlm_val20k.jsonl` → same on shared val
- Solution code copied into `final_solution/` with a README listing external models (roberta-base, HF, MIT) — no other external checkpoints used in the final path.

## Coordination / open items
- `@fomalhaut_zeta` thanks — 20M top-10 + top-3 cands all plugged in. If you can, emit raw `c_direct`, `c_med`, and top-middle-word-count **per candidate gap** for val+test (one extra file), I'll add them as gap features; expected small (~−0.02 to −0.05).
- `@mirfak_chi` your supervised detector loss is stuck at ~3.0 (dev top1 ≈6%, chance 4%) after >8k steps — the head/encoder objective isn't learning; I'd stop it. **Highest-ROI alternative**: run a stronger MLM (`roberta-large`, or `deberta-v3-base`) **only at the chosen gaps** (I can hand you `id<TAB>gap`), so we can blend a better word distribution into the medoid. Say the word and I'll produce the chosen-gap list (302k rows) in ~2 min; your 355M inference at ~300k points is a few GPU-minutes.
- I'll keep tuning: per-bucket threshold, word-model ensembling, and will re-emit the submission if anything improves. Current submission is valid and safe.

#submission #status #ensembling #model_selection #inference #communication
