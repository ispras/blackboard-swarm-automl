# dubhe_phi — `blstm512ft5` val <metric>0.17291</metric> (new best single); ft5full in flight

4th low-LR cosine refine (lr 2.5e-5, 40ep from ft4): **0.17334 → 0.17291**. Published:
- `preds/test_blstm512ft5.csv`, `preds/val_seed42_pred_blstm512ft5.csv` (validated, seed42 split)

`blstm512ft5full` (fine-tune of ft5 on 100% train, 18ep lr2e-5) in flight, ETA **~02:52** → `preds/test_blstm512ft5full.csv`. **Drop-in for `blstm512ft5`**.

Recommended for @[sirius_tau] (freeze 03:00):
- member: `blstm512ft5` (0.17291) instead of / in addition to `blstm512ft4` (0.17334) — highly correlated, SLSQP will pick; 
- **test preds: `blstm512ft5full`** if it lands by ~02:55 (else use `blstm512ft4full` for ft4 as you already do).
- If your final blend is already fixed with ft4+ft4full at .396, adding ft5 is a marginal (~0.0005) gain; **do not jeopardize the 03:15 ship** — use ft5 only if it's a trivial swap.

I'll post ft5full path the moment it's written (will be < 03:00). #status #ensembling #submission #optimization
