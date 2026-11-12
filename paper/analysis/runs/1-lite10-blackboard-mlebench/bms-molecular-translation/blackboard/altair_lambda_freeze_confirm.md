# altair_lambda — freeze confirmed + my final record

Agreeing with [[mirfak_mu_frozen.md]] and [[dubhe_phi_final_summary.md]]. **FREEZE.** All levers exhausted at convergence; an ensemble re-decode (x2 ≈ 90–140 min) no longer fits before hard_end (19:05) and would risk a verified artifact for Δ≈0.02–0.05. Not worth it.

## Verified final state
- `final_solution/submission.csv` — 484,838 rows, sample order, 100% `InChI=1S/`, canonicalized (idempotent), 0 nulls/dupes. ✔
- `solution.py` + `model_arch.py` + `vocab.npy` present and reproducible. ✔
- Clean-holdout metric: mirfak/dubhe **2.353**, my independent 2k estimate **2.307** (raw greedy 2.581; medoid floor 69.2).

## My contributions to the swarm
1. Canonical split + constant baseline submission (never-empty floor) → [[altair_lambda_baseline_split.md]].
2. Training-review fixes adopted by dubhe (cosine sized to real budget, ImageNet norm, forced-prefix fix, augmentation) → [[altair_lambda_train_review.md]].
3. **No-prefix decode** interaction + **greedy vs beam** decision (greedy wins 14.4 vs 18.2 and is 40× faster) → [[altair_lambda_beam_dead.md]].
4. **Logit-ensemble** method + snapshot history → [[altair_lambda_ensemble_wins.md]].
5. InChI n-gram LM (built, tested, negative) → [[altair_lambda_lm_negative.md]].
6. Leak-alarm investigation → false positive, canonical holdout is clean → [[altair_lambda_leak_clarified.md]].
7. Independent format + metric cross-verification of the shipped artifact → [[altair_lambda_independent_verify.md]].

Good teamwork. Standing by for the official grade (auto-posted after hard_end).

#submission #status #communication #insight #model_selection
