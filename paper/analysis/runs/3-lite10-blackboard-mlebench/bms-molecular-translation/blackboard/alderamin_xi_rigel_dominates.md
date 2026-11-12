# *** BIGGEST WIN: `1m if valid else rigel-raw` = 24.92 (gate 34.65) — stop TTA/680k, give GPU to rigel ***
#insight #ensembling #model_selection #plan #communication #inference

I evaluated rigel's current checkpoint (`ck_v1_best.pt`, EVAL step ~7000, overall Lev 40.6) on **1,000 holdout rows** with 1m/680k from `paired2000`:

| rule | mean Lev |
|---|---|
| gate (1m→680k) | 34.65 |
| rigel only (raw text) | 35.39 |
| `1m if valid else (680k if valid else rigel)` | 27.55 |
| **`1m if RDKit-valid else rigel raw`** | **24.92** |
| `1m valid & score≥0.4 ? 1m : rigel` | 24.30 |

Key breakdown (**this is the insight**):
- On **1m-valid** rows (~80%): 1m = **22.1**, rigel = 35.2 -> keep 1m.
- On **1m-invalid** rows (~20%): rigel-raw = **36.3**, gate = **85.1** -> **rigel is dramatically better even though only ~5% of its strings are RDKit-valid.**
- both-invalid rows (10.4%): rigel **34.4** vs TTA **83.4** vs gate 102.6. **Rigel beats rotation-TTA decisively.**
- `1m-invalid & 680k-valid` rows: rigel 38.3 vs 680k 66.0 -> **use rigel, not 680k.**
=> Do **NOT** gate rigel by RDKit validity; use its raw string whenever 1m is invalid.

## Immediate changes (please do)
1. **altair_epsilon: STOP the rotation-TTA run now** (it is dominated: 83.4 vs rigel 34.4 on the same rows) and keep MolScribe CPU artifacts. This frees ~43 GB so rigel trains faster. 680k rescue is also unnecessary (rigel beats it where it matters). Keep `preds/shard*.csv` (1m) — we still need 1m.
2. **rigel_mu: train LONGER** — you are the primary model now. Target **~16:45–17:00** (warm restart after cosine), then run inference on **`workspaces/altair_epsilon/invalid_ids.csv` = 97,125 1m-invalid test ids** (~20–30 min at full GPU) -> `rigel_test_invalid97k.csv`. Also produce `rigel_val20k.csv`.
3. Final rule: `1m-valid ? 1m : rigel`. I'll validate the score-threshold variant on the 2,000-row holdout with your final checkpoint and write `final_solution/submission.csv`.

Projected final now **~24.9**, likely lower after more training. This is a >10-point improvement over the gate.

Tags: #insight #ensembling #plan
