# dubhe_phi — FINAL CONFIRMATION (final solution in place)

Per SYSTEM final-phase directive. No new experiments running (384 probe killed earlier; GPU idle).

## `final_solution/` — verified
- `submission.csv` — **484,838 rows**, cols `image_id,InChI`, ids **exactly in `sample_submission` order**, 0 nulls, **100% `InChI=1S/`**, RDKit-canonicalized. md5 `9f7aacc69646348277e6fe39eef3955a` (unchanged since freeze).
- `solution.py` (reproducible decode: greedy no_prefix, max_len420, size-sorted + RDKit canon), `model_arch.py`, `vocab.npy`.
- Model ckpt: `workspaces/dubhe_phi/run1_best.pt` (= `workspaces/mirfak_mu/ckpts/best_final.pt`).

## Model & metric
ConvNeXt-Tiny (ImageNet) encoder @320 + 6-layer Transformer char decoder, trained on 1,919,348 images (20,000 canonical holdout). Greedy free generation (no forced prefix) + RDKit canonicalization.
**Clean-holdout mean Levenshtein = <metric>2.353</metric>** (77% exact; independent cross-checks 2.307 / 2.545; raw greedy 2.581; raw-greedy floor 69.2).

Frozen and agreed by all three agents ([[mirfak_mu_frozen.md]], [[altair_lambda_freeze_confirm.md]], [[dubhe_phi_final_summary.md]]).

#submission #status #communication
