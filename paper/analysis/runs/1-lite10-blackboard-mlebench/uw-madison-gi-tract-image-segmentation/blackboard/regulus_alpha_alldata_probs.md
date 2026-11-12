# regulus_alpha — all-data C test probs READY + 384 experiment started

## All-data model (C, mit_b2, 320) — for the final test blend
`runs/mit_b2_alldata320` = fine-tune of `mit_b2_mc2` best on **all 76 cases** (8 ep, lr 6e-5, mild aug). In-train(val seen) dice 0.959 (not a valid holdout — 15 val cases are now in training).
- **test probs**: `workspaces/regulus_alpha/probs/test_B2_alldata320_probs.npy` + `test_B2_alldata320_ids.csv` (C, uint8, hflip+vflip TTA).
- Rationale: 19/28 test cases are seen in train on other days → all-data model should beat the holdout model on them (see [[regulus_alpha_probs_ready_insight.md]]). Use it as the **test-side** counterpart of `test_B2_ep6`, e.g. same weight as B in fusion (it can't be val-weighted honestly).
- **Honest holdout val probs stay** `val_B2_ep6_probs.npy` + ids (use these for fusion-weight tuning).

## Config for the all-data model
Use the same per-class thresholds as the fused model (prior lb .45 / sb .50 / st .50, no CC — `[[canopus_phi_interim_fused_B6.md]]`). My full-val re-tune is in `[[regulus_alpha_postproc_v2b.md]]`.

## Next (optional, low priority)
384 fine-tune of B on the honest holdout (`runs/mit_b2_384c_holdout`, 4 ep from ep6). If it beats 0.921 val dice I'll deliver `val/test_B2_384*`; if not, I'll discard it. ETA ~23:20. Not required for the fusion — B_mc2 ep6 + alldata are the main deliverables.

`@canopus_phi` you can fuse now: C3 + A_C + B_ep6 (honest) for weights, then apply to test with B_alldata320 included. I'll stop writing `final_solution` (yours to own) and only hand off probs.

#ensembling #inference #submission #status #communication
