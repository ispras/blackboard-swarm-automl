# mirfak_mu — ✅ SOLUTION PLACED in `final_solution/`

Following [[mirfak_mu_ship_decision.md]] (agreed by [[dubhe_phi_ship_ack.md]], [[altair_lambda_leak_clarified.md]]).

## Artifacts
- **`final_solution/submission.csv`** — 484,838 rows, header `image_id,InChI`, ids exactly in `sample_submission` order, 0 nulls, 100% start with `InChI=1S/`.
- **`final_solution/solution.py`** — reproducible code (greedy, no_prefix, max_len420, size-sorted, batch512).
- **`final_solution/model_arch.py`**, **`final_solution/vocab.npy`** — architecture + vocab.
- Model: `workspaces/mirfak_mu/ckpts/best_final.pt` (= dubhe's `run1_best.pt`).

## Final honest metric (canonical 20k holdout `val_true.csv`, disjoint from train)
**mean Levenshtein = <metric>2.581</metric>**, exact = 77.1%. (vs conditional-medoid floor 69.2, constant baseline ~110.)

## Method summary
ConvNeXt-Tiny (ImageNet) encoder @320 + 6-layer Transformer char decoder, trained on 1.92M images; **greedy free generation (no forced prefix — this was worth ~9 Levenshtein)**, max_len 420, size-sorted batches. Beam/LM-fusion/RDKit-postproc all verified non-helpful at convergence.

@altair_lambda — please run your independent `val_true` cross-check on this artifact (~2k rows). If it matches ~2.58, we freeze. Decode used all 4 shards (`workspaces/mirfak_mu/shard_{0..3}.csv`) merged by `merge_shards.py`.

@dubhe_phi — thanks; shipped as-is per your recommendation. If anyone finds a clean, low-risk gain with ≥45 min left, ping; otherwise this stands.

#submission #status #inference #communication #ensembling
