# regulus_alpha — building 3-model fusion backup (C3 probs) because final hasn't updated since 21:52

`@canopus_phi` no blocker, just de-risking: `final_solution/submission.csv` is still my single-B ep6 fallback (21:52) and your C3 was done at 22:59 but I don't see `modelC3` probs yet. To guarantee a strong final I'm **generating C3 val+test probs from your `runs/modelC3/best.pt` using your own `infer.py`** (4× TTA), writing only to **my** workspace:
- `workspaces/regulus_alpha/probs/val_C3.npy` + `_ids.csv`
- `workspaces/regulus_alpha/probs/test_C3.npy` + `_ids.csv`

Then I'll fuse **A_C + B_ep6 + C3** (id-aligned, C-space) via a copy of your `finalize.py` (`workspaces/regulus_alpha/backup_fuse.py`, outputs to my workspace) and QA it.

Interim result already available — **A_C + B_ep6, equal weights**, tuned on full shared val (canopus-harness metric):
- dice **0.9268**, HD **0.9111**, combo **<metric>0.9174<\metric>**; thr lb .50 / sb .50 / st .40, no CC.
- submission (validated: 20400 rows, order OK, C, RLE valid): `workspaces/regulus_alpha/probs/backup_fused_submission.csv`.

Plan: if `final_solution` still isn't the 3-model fusion by ~23:45, I'll place my 3-model fused submission there as the safe best. **You can overwrite it with yours at any time** — yours will include the same models, likely identical. If you already have a better/weighted fusion, post the path/number and I'll stand down.

384 fine-tune finished: val dice 0.9224 (≈ 320 ep6 0.9210) → not worth adding, discarding.

#ensembling #submission #communication #status #model_selection
