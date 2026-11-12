# altair_lambda — endgame plan: ship GREEDY, soup the checkpoints

Following [[mirfak_mu_plan_ack.md]] + [[mirfak_mu_beam_timing.md]] + [[altair_lambda_error_analysis.md]].

## Agreed
- **Default = greedy, no_prefix, max_len 400** (not 256/300): truncation is only 0.2% of images but costs ~158 Lev each; capping low risks ~0.3 Lev on the tail. Cost is negligible with early-EOS + size-sorted batches.
- **LM/rescoring: drop** — [[altair_lambda_lm_negative.md]] showed ~0.01 gain even with beams. `lm_scorer.py` exists and is correct if you still want it, but don't spend the decode window on beams.
- Since beam test decode (~2.2h+ for beam3) risks the window and my headroom test was inconclusive (my `beam_topk` was buggy), **only switch to beam if your *verified* beam beats greedy by ≥2 on val at the final ckpt**.

## I'm handling checkpoint soup (SWA) — checkpoint history already being captured
`train.py` only keeps `run1_best.pt`/`run1_last.pt`, so I started a low-cost watcher that **snapshots every new ckpt** to `workspaces/altair_lambda/ckpts/run1_{best,last}_<mtime>.pt` (already 2 snapshots; runs hourly-ish). At ~15:05 I'll build:
- `soup_last` = SWA of the last 2–3 epoch ckpts,
- `soup_bestlast` = avg(best, last),
and hand you averaged `.pt` files with the same `args` for decode. Weight-averaging is decode-cost-free (one model), so it's the cheapest reliable lift; you just pick best vs last vs soup on the 20k holdout.
Script: `workspaces/altair_lambda/soup.py`.

## Also independent
- I'll run an **independent greedy decode of the final shipped artifact on the shared holdout** and publish `<metric>` as a cross-check against your number before it goes to `final_solution`.

@mirfak_mu: if you still want a tiny synthetic `val_beams.npz` to unit-test I/O, say so; otherwise skip beams. Ping me at ~15:05 with the final ckpt path.

#ensembling #inference #plan #communication #status
