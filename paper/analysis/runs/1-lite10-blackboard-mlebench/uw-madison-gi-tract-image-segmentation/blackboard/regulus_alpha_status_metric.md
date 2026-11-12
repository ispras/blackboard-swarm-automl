# regulus_alpha status — Model B training + local metric harness + ensemble protocol

## Model B (mit_b2, 2.5D, 320) — running
`workspaces/regulus_alpha/runs/mit_b2_s42/` (best.pt saved each improved epoch).
- ep1/36: loss 0.5885, **val_dice 0.5566** (empty-empty=1), positive-only 0.1345, per-class [lb .449, sb .727, st .493]. Epoch time 353s with 3 concurrent trainings → ETA ~3–3.5h.
- Split = shared 15-case holdout `[[regulus_alpha_cache_and_gpu.md]]` (`workspaces/regulus_alpha/val_cases.json`).

## Local metric harness (reusable): `workspaces/regulus_alpha/metric.py`
- `rle_encode` / `rle_decode` — **verified exact round-trip on 3000 train rows** (column-major `order='F'`; matches Kaggle spec, handles non-square 360×310 correctly).
- `dice_slice(..., empty_empty=1.0 or 0.0)` — both variants, so we can check the TASK caveat.
- `hd3d` — normalized 3D Hausdorff (coords /Z,/H,/W; bbox-cropped for speed ~0.1–0.5s). **Confirmed far false-positive costs ~0.56 HD** → postproc is high-value.
- `eval_postproc.py` — threshold / small-component / fill-holes / 3D-largest-CC sweep on holdout probs.
- `make_submission.py` — chunked ensemble (weighted avg) + per-case-day 3D CC + 2D CC + resize-to-original + RLE; writes in `test.csv` order.

## Ensemble protocol (important: orders differ!)
My test/train row order ≠ `[[canopus_phi_cache_ready.md]]` order (same id set, different order). **Do NOT fuse positionally** — ship a `*_ids.csv` next to each `*_probs.npy` and align by `id`. My `infer.py` already writes `<out>_probs.npy` + `<out>_ids.csv`.
I'll produce val+test uint8 probs with hflip+vflip TTA when training finishes.

`@canopus_phi @aldebaran_omicron` — if you want, I can own the **postprocessing study** (I already have the harness) and give you the winning config; you own fusion. Or I can fuse. Reply preference.

#status #postprocessing #metric #ensembling #communication #inference
