# rigel_mu — ACK 283k ids; schedule: stop training ~16:00, deliver holdout first then 283k

Refs: [[alderamin_xi_rigel_infer_ids_ready.md]] [[rigel_mu_val_25.md]]

## ACK
- I'll use **`workspaces/alderamin_xi/rigel_infer_ids.csv` (283,168)** — good, covers thresholds 0.4–0.7. If I'm short on time I'll cut to the 97k-invalid subset and tell you; but 283k should fit.
- Val: 25.9@12k, 27.2@13k (noise; LR still 1.8e-4 and annealing to ~0 by ~16:50).

## Faster inference implemented
I rewrote greedy decoding with **preallocated KV caches** (`model.generate_fast`, no per-step `torch.cat`): **56 img/s @bs384 even while training runs** (would be ~42 before; will be higher once training stops). Quality unchanged (Lev 25.09 on 768 vs 25.9 on 1024). ~283k @ ~65–75 img/s alone => **~65–75 min**.

## Schedule (deliberate, for safety)
- **Train until ~16:00**, then **stop** (no later) so inference + assembly have ≥1h buffer before hard_end.
- **16:00–16:08**: infer on **full `analysis/holdout_20k.csv`** -> `rigel_val20k.csv`, post path immediately so you can tune the threshold while I run test.
- **16:08–17:20**: infer on **283,168** ids -> `rigel_test_283k.csv` (image_id,InChI; raw strings).
- If the 283k run is too slow, I'll publish partial shards; but expect full.
- All from the **best checkpoint by holdout Lev** (`ck_v1_best.pt`), same model for val and test.

If you prefer I **train to 16:40** instead, say so now and I'll shift (but then 283k may not finish before ~18:00).

Tags: #inference #plan #status #communication
