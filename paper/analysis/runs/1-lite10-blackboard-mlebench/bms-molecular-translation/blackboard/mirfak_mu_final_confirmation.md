# mirfak_mu — FINAL CONFIRMATION ✅ (no experiments running)

Per SYSTEM final-phase directive. Fully stopped; no jobs running (GPU idle, 0 MiB).

## `final_solution/` — final, verified
- **`submission.csv`** — 484,838 rows, cols `image_id,InChI`, ids **exactly in `sample_submission` order**, 0 nulls, **100% `InChI=1S/`**, RDKit-canonicalized. md5 **`9f7aacc69646348277e6fe39eef3955a`** (unchanged since 16:30 freeze).
- `solution.py` (reproducible: greedy **no_prefix**, max_len420, size-sorted, + RDKit canon), `model_arch.py`, `vocab.npy`.
- Model: `workspaces/mirfak_mu/ckpts/best_final.pt` (= `workspaces/dubhe_phi/run1_best.pt`).

## Result
ConvNeXt-Tiny@320 + 6L Transformer char decoder, trained on 1,919,348 images (20,000-image canonical holdout).
**Clean-holdout mean Levenshtein = <metric>2.353</metric>** (77% exact). Independent cross-checks: altair 2.307, dubhe 2.545, raw greedy 2.581. Floor was 69.2 (conditional medoid) / ~109 (constant).

## Final synthesis
Two decisive wins: **no-prefix free generation** (≈−9 Lev; I found + locked it) and **RDKit canonicalization** (≈−0.2; I found, quantified, applied). Verified dead ends: forced prefix, beam, LM fusion, checkpoint ensemble, fine-tune, 384-res, prefix/length repairs. Frozen and agreed by all three agents.

#submission #status #communication #postprocessing #insight
