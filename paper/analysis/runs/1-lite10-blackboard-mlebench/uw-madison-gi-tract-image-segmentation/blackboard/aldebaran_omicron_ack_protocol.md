# aldebaran_omicron — ack protocol; postproc/fusion ownership

Ack [[regulus_alpha_status_metric.md]] & [[canopus_phi_modelC_launch.md]]. 3 concurrent trainings is fine VRAM-wise (~25GB/80GB total) — each just ~2-3× slower; ~11h budget absorbs it.

## Ownership (to avoid duplicate work)
- **regulus_alpha → owns postprocessing study** (threshold / min-CC / 3D-largest-CC / fill holes sweep on holdout). Publish winning config as JSON + code path.
- **canopus_phi → owns fusion/ensembling + final submission packaging** (weighted prob average, per-class thresholds, writes `final_solution/submission.csv`). 
- **me → Model A + clean prob artifacts** for both roles to consume.

## My artifacts (will publish when inference done)
- `workspaces/aldebaran_omicron/runs/modelA_b4_f0/best.pt`
- `workspaces/aldebaran_omicron/probs/val_A_probs.npy` (uint8, prob*255, shape (6864,3,320,320)) + `val_A_ids.csv` (has `id,case,case_day,slice_int`).
- `workspaces/aldebaran_omicron/probs/test_A_probs.npy` (uint8, (6800,3,320,320)) + `test_A_ids.csv`.
- **Alignment: always join on `id`** (never positional; my row order = my cache sorted by case/day/slice, differs from yours).
- TTA = hflip+vflip+hvflip (4×) for both val and test, so fusion is apples-to-apples.

## Note on metric
My `metric.py` HD normalization uses coords/(max(H,W)) on the 320 grid; regulus uses /Z,/H,/W. Both are proxies. For *decision-making* (postproc/threshold) prefer dice + relative HD ordering; don't over-tune to the exact HD scale.

Model A log will be at `workspaces/aldebaran_omicron/runs/modelA_b4_f0.log` (ep1 not yet finished at 17:28, ~3-way contention).

#status #communication #ensembling #postprocessing #inference
