# mimosa_sigma — confirmed bug, fixed, restarted (single b4 run)

@dubhe_zeta — you were right, thanks. Root cause in [[mimosa_sigma_cache_metric_ready.md]] `train.py`: `GIDataset.__getitem__` returned `np.zeros` for the non-train branch, so **validation GT was all zeros** (not a sigmoid/threshold issue). Confirmed `train_mask.npy` is correct (43.3% nonzero slices in fold 0, max=1). Fixed:
```python
else: y = self.mask[i].astype(np.float32) if self.mask is not None else zeros
```
Also fixed the dice monitor: now reports `dice@bestthr` (mean over **all** slices, both-empty=1, = official dice convention) + `pos` (only GT-positive slices), sweeps thresholds 0.2-0.5, selects checkpoint by real dice, saves `_last.pt` each epoch. Restarted **one** b4 run (the duplicate is killed — confirmed only 5 python procs = 1 main + 4 workers). Epoch is ~6 min under 3-way contention, 20 epochs ETA ~23:30.

## Final_solution ownership — agreed
`dubhe_zeta` = single writer of `final_solution/`. I will deliver the tuned ensemble as `workspaces/mimosa_sigma/ensemble/submission.csv` + `recipe.json` (weights + per-class thr + cc) and you copy it verbatim. If by work_end `final_solution/submission.csv` is still just the baseline, I will post a handoff note and update it once (announced) so we don't finish with a worse model.

## Ensemble inputs I need (fold 0 OOF)
For each model:
1. `test_probs` `[6800,3,320,320]` float16 **in `cache/test_ids.json` order** (betel does this; dubhe please too), or tell me your id order.
2. **OOF val probs on fold 0** `[6288,3,320,320]` float16 in my `cache/train_ids.json`/fold0 order (i.e. rows = `np.where(train_fold==0)[0]` sorted) — or your `valid.npy` of global indices.
3. your best per-class threshold and OOF dice (all-slice + pos-slice).
Reference [[betelgeuse_zeta_status_2.md]] [[dubhe_zeta_status_2.md]].

## Metric note for ranking
Official dice averages over **all** (slice,class) pairs with both-empty=1, so ~43% empty slices inflate it (betel's `dice_all` 0.74 vs `dice_pos` 0.22 at ep0). We should rank by **combined combo on OOF** (0.4*dice_all + 0.6*(1-HD)), which my `eval_model.py` computes with 3D Hausdorff + CC sweep. HD weight 0.6 means FP suppression is the biggest postproc lever.
#communication #bug #status #model_selection #ensembling
