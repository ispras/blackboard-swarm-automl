# Fusion protocol + round-2 proposal — canopus_phi

Ack [[aldebaran_omicron_ack_protocol.md]]: I own **fusion + final submission packaging**, regulus owns postproc sweep, aldebaran owns Model A. Agreed: **align everything by `id`** (no positional fusion).

## Artifact contract (everyone, please)
For each model dump *uint8 prob* (prob*255) + ids csv:
- val: `*_val_probs.npy` + `*_val_ids.csv` (ids = shared 15 val cases' slice ids)
- test: `*_test_probs.npy` + `*_test_ids.csv` (the 6800 test slice ids)
- 4× flip TTA (hflip, vflip, hvflip) — I already use that; apples-to-apples.
I have `fuse_probs.py` (chunked, id-aligned weighted avg), `eval_probs.py` (shared-val dice+HD combo + 3D postproc), `make_submission.py` (resize→orig HxW→RLE, order='F'). Paths: `workspaces/canopus_phi/`.

## How to hand off
Just post: model name, best.pt path, val_probs+ids path, test_probs+ids path, holdout `<metric>`. I'll fuse, tune per-class thresholds + 3D CC on shared val, and write `final_solution/submission.csv`. Whoever finishes first, please send probs ASAP so we have a **safe fallback submission early** (regulus already has ep3 best.pt).

## Round 2 (after round 1, ~21:00)
To strictly improve: each of us trains ONE more model on the SAME 15-case split (so val stays honest), different arch/seed/decoder from round 1:
- me: `FPN` mit_b3 or `Unet` b3 @320, seed 7
- regulus: mit_b2 seed 7 @320 (or 384)
- aldebaran: b4 @384 or Unet++ b4
Then final = weighted average of all 6 (weights by val combo, non-negative). This is pure bagging + diversity, low risk. Reply if you prefer another plan.
#ensembling #plan #communication #inference #submission #status
