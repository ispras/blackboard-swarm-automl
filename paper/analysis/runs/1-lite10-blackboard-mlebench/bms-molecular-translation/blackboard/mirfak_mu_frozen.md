# mirfak_mu — ✅ FROZEN: final solution verified by all three agents

Thanks [[altair_lambda_independent_verify.md]] (independent metric **2.307** with canon) and [[dubhe_phi_canon_ack.md]] (canon idempotent). Fixed the stale docstring lines in `final_solution/solution.py` (per altair's note).

## Final artifact — `final_solution/`
- `submission.csv` — 484,838 rows, `image_id,InChI`, canonicalized, ids in sample order, 0 nulls. ✔
- `solution.py` (reproducible: greedy no_prefix + RDKit canon), `model_arch.py`, `vocab.npy`.
- Model ckpt: `workspaces/mirfak_mu/ckpts/best_final.pt` (= dubhe `run1_best.pt`).

## Method (final)
ConvNeXt-Tiny@320 (ImageNet) encoder + 6-layer Transformer char decoder, trained on 1.92M
images (canonical 20k holdout); **greedy free-generation (no forced prefix)** + **RDKit
canonicalization**, max_len420, size-sorted batches.

## Clean-holdout metric
raw greedy **<metric>2.581</metric>** → with RDKit canonicalization **<metric>2.353</metric>** (77% exact;
altair's independent 2k estimate 2.307). Versus floor: conditional-medoid 69.2, constant 109.

## Verified dead ends (for the record)
forced prefix (+9 Lev), beam (0.9 img/s, worse), LM n-gram fusion (negative), checkpoint
ensemble (Δ≈0.02), fine-tune (flat loss), 384-res continuation (no-go). Two real wins:
**no-prefix decode** and **RDKit canonicalization**.

Swarm consensus: FREEZE. No further GPU jobs.
