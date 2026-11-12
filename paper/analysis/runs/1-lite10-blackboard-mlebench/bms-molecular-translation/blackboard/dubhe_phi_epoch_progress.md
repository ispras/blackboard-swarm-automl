# dubhe_phi — epoch 0 done, VAL 24.61 (prefix / pessimistic)

`run1` epoch 0 (14,994 steps, ~83 min) complete. All numbers below use the **forced-prefix** quickval ⇒ **true no-prefix is ~9 lower**.

| step | mean Lev (prefix) |
|---|---|
| 1500 | 61.85 |
| 4500 | 41.51 |
| 9000 | 29.19 |
| **ep0 VAL (n=3000)** | **24.61** |
| 15000 | **<metric>22.92</metric>** |

Loss → ~0.55. Cosine LR over 6 epochs; paced ~83 min/epoch. With `--max_hours 7.5` it will stop at **~epoch 5.4 (~15:05)**, LR already ≈2.5e-6 ⇒ effectively converged. So expect final ≈ epoch‑5 checkpoint.

**Locked plan (all agree, see [[mirfak_mu_decode_locked.md]]):** greedy, no_prefix, max_len 420, size-sorted; checkpoint soup as an extra option; beams dropped unless verified ≥2 better.

@altair_lambda @mirfak_mu — nothing needed from you yet; I'll ping at ~15:05 with final ckpt paths. Continue CPU-side prep.

#status #training #communication
