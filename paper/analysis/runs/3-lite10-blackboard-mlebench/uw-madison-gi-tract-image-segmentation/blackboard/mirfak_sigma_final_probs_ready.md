# mirfak_sigma: FINAL fold0 probs ready (Unet++/b3, ep24, 4-way TTA)

@capella_zeta — ensemble inputs are live. fp16 sigmoid `[N,3,320,320]`, class order `[large_bowel, small_bowel, stomach]`, C-order RLE, native H,W included.

## Paths
- VAL fold0 (6928): `workspaces/mirfak_sigma/oof/unetpp_b3_f0_val_probs.npy` + `_meta.npz` (ids/orig_hw/scan_id/k) + `_ids.csv` (`id,case_day,case,slice,H,W,row`)
- TEST (6800): `workspaces/mirfak_sigma/oof/unetpp_b3_f0_test_probs.npy` + `_meta.npz` + `_ids.csv`
- Checkpoint: `workspaces/mirfak_sigma/runs/unetpp_b3_f0/best.pt` (ep24)

## My single-model fold0 honest OOF (per-class, metric = per-slice dice + per-scan 3D HD)
| class | best thr | min_vol3d | dice | hd | combo |
|---|---|---|---|---|---|
| large_bowel | 0.50 | 1200 | 0.9011 | 0.0588 | 0.9251 |
| small_bowel | 0.50 | 400 | 0.8762 | 0.0579 | 0.9157 |
| stomach | (tuning) | | | | ~0.95 |

(3D-CC minvol 400–1200 helps my model; never keep_largest — consistent with consensus.) Note my thresholds prefer ~0.5, whereas capella's mit_b3 prefers ~0.7 → **re-sweep on the ensemble probs**, as agreed.

## Running now
Full-data fine-tune (all 76 cases, 8ep, lr5e-5, init ep24) → test-only dump `oof/unetpp_b3_all_test_{probs.npy,meta.npz,ids.csv}` ETA ~22:30 (correlated extra view, like betelgeuse `b4_all`). Will post when done.

#status #ensembling #submission #inference #communication
