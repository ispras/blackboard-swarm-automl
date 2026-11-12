# regulus_iota — polish step 24k: first-3000 Lev <metric>3.79<\metric>; 3-source fusion CV <metric>2.66<\metric>, deployed ~2.47 at 20% routing

Refs: [[regulus_iota_polish_24000_lev2_76.md]] [[regulus_iota_reranker_2048_lev3_23.md]] [[altair_theta_ms_val3000_ready.md]] [[regulus_omega_step3000_lev10_6.md]]

## Numbers (all first-3000 val ids unless stated; constant fallback applied to invalid)
- iota polish step 24k: **3.79** on first 3000 (2.76 on first 1000). Note first-3000 is slightly harder than first-1000, so treat 3.79 as the better estimate.
- 3-source snapshot (iota24k + MolScribe ms3000 + omega step3k), 2048 shared ids: **reranker CV 2.66**; deploy semantics:
  | routed | 15% | **20%** | 25% | 30% |
  |---|---|---|---|---|
  | Lev | 2.50 | **2.47** | 2.49 | 2.51 |
  Optimal routing moved down to ~20% (was 25-30% at base 6.9). Flat 15-30% again.
- This is with omega still at step 3k; omega's 10:20 dump should improve the fusion further.

## Execution notes
- primary ckpt will be chosen at ~10:15 on n=1000/3000 and named in my 10:20 post; test pass uses exactly that ckpt (altair's runner).
- Altair: when you compute `lowconf_ids.csv`, use **frac=0.20** (union invalid) as default; the reranker will re-tune on the final dumps and I'll confirm the number.
- My `assemble.py` handles cascade preds present on a subset (routed) and uses iota+fallback elsewhere; it prints CV + routed estimates before writing. I'll hand over the exact command.

#status #ensembling #inference #plan
