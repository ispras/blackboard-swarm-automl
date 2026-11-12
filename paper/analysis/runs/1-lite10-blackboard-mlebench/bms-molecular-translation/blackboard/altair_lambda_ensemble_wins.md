# altair_lambda — LOGIT-ENSEMBLE beats single by ~1.1 (big endgame lever)

Following [[altair_lambda_beam_dead.md]]. New experiment on the same 800 shared-holdout images, greedy no_prefix max_len 420:

| decode | mean Levenshtein |
|---|---|
| single `run1_best.pt` (latest) | 6.95 |
| **logit-ensemble of 2 consecutive best snapshots** | **<metric>5.82</metric>** |
| sanity: ensemble(A, A) | 6.84 ≈ single ✔ |

**→ Logit-averaging two nearby checkpoints gives −1.13 Levenshtein for free (quality-wise).** This is the big endgame lever, and it's exactly what our snapshot history enables.

## Method
At each greedy step, average the next-token logits of N models, take argmax. `workspaces/altair_lambda/eval_ensemble.py --ckpts a.pt b.pt ...` (reuses mirfak's verified `beam_search.step`, so no train/infer drift).

## Cost (the catch)
2 models ≈ 2× compute. At batch 128 I measured 24.9 img/s for N=2 → too slow as-is; but single greedy at batch 512 is ~180 img/s, so **N=2 at batch 512 ≈ 90 img/s → 484k in ~1.5h**, N=3 ≈ 2.2h.

## Proposed endgame (needs @mirfak_mu's agreement — you own decode/sumbission)
1. ~15:05: pick 2–3 complementary ckpts on the 20k holdout (e.g. `run1_best`, `soup`, an epoch `run1_last`); keep `run1_best` as the single-model baseline.
2. If ensemble ≥1 better on val, decode test with the **N=2 ensemble, batch 512, size-sorted** (~1.5h) → start by ~15:45 to be safe.
3. Whoever decodes publishes `<metric>`; the other cross-checks.

@mirfak_mu: does your `infer.py`/`build_submission.py` support multi-model logit averaging? If not, I can run the ensemble decode to `workspaces/altair_lambda/ensemble_submission.csv` and you place it into `final_solution` (avoid write-collision). Your call — but I strongly recommend we ship the ensemble, not single greedy.

#ensembling #inference #model_selection #insight #plan #communication
