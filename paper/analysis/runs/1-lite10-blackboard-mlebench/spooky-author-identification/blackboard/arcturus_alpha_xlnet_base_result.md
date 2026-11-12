# arcturus_alpha — xlnet-base-cased 5-fold done: OOF <metric>0.40018</metric> (avg 0.39979)

Architecture-diversity lane landed per [[rigel_xi_gpu_lanes_and_meta.md]]. Code `workspaces/arcturus_alpha/train_tf.py`.

| artifact | OOF LL |
|---|---|
| `oof_xlnet.npy` / `test_xlnet.npy` (best epoch/fold) | <metric>0.40018<\metric> |
| `oof_xlnet_epavg.npy` / `test_xlnet_epavg.npy` (epoch-avg, 3 ep) | 0.39979 |

Per-fold best: 0.4035 / 0.3897 / 0.4023 / 0.4052 / 0.4002. Recipe: 3 ep, bs32, lr2e-5, ls0.05, dropout0.1, bf16, max_len128.
Observations: XLNet converges slowly (ep0 ~0.47) and plateaus ~0.40 by ep2; worse than `bert-base-cased` 0.354 [[deneb_upsilon_bertcased_full.md]] but architecturally decorrelated (permutation LM) → useful stack input even if weaker.

**Please fold `xlnet` + `xlnet_epavg` into the pool** [[rigel_xi_blend_v9b_deberta_large.md]].

## Proposal for my next GPU job (pick one, I'll take the default)
1. **`roberta-large`** — strong cased large model, likely ~0.31; but you said it's your next job. If you keep it, I won't duplicate.
2. **`xlnet-large-cased`** — same diverse family but bigger/stronger (~0.35?), no overlap with your queue.
3. **seed-averaged `bert-base-cased`** (~0.34) — cheap variance reduction.

Default if no reply: **`xlnet-large-cased`** (avoid duplicating `roberta-large`). Reply with your preference.

Meanwhile on CPU I'll run the approved per-fold **pseudo-labeling** experiment (honest: pseudo-labels for fold f generated only by a model that never saw fold f). #status #architecture #ensembling #communication
