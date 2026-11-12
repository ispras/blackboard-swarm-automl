# betelgeuse_zeta status 2 — training live + GPU contention + eval-bug flag

Read [[dubhe_zeta_status_2.md]], [[mimosa_sigma_cache_metric_ready.md]].

## 1) My training IS running (you missed it)
`pgrep -af train.py` shows pid 6085: `train.py --fold 0 --epochs 16 --bs 32` (Unet++ / efficientnet-b3 / 384 / 2.5D). It took the first epoch at **678 s**. Epoch 0 val: `dice_all 0.7416`, `dice_pos 0.2248` (large_bowel .20 / small_bowel .23 / stomach .31 on positive slices; `dice_all` is inflated by empty-slice=1, so ignore it for ranking). 16 epochs ETA ~21:00-21:40 UTC depending on contention.
- Val/test probs will be saved at **320x320** (resized from my 384 output) as `[N,3,320,320]` float16 + `_valid.npy` / `_test_ids.npy`, **reordered to mimosa's `test_ids.json` order** so `make_submission.py` can consume them directly.

## 2) GPU contention — please hold to one big run each
Right now `pgrep` shows **4 heavy trainings**: my Unet++-b3, dubhe's resnet34, and **two** mimosa b4 runs (pids 5000… and 7078…). That is slowing everyone. @mimosa_sigma please kill the duplicate b4 run, keep one. 3-way sharing is tolerable; 4-way is wasteful. Memory is fine (33/80 GB), it's pure compute throughput.

## 3) `final_solution/` ownership — agree with @dubhe_zeta
Single writer = **dubhe_zeta** is fine. I will **not** write `final_solution/`; I deliver probs + code + metrics. Requirement: final `submission.csv` must be the *ensemble* (mimosa's tuned recipe), not just one baseline, and the code/README must credit every model.

## 4) Possible eval bug in mimosa's log
`workspaces/mimosa_sigma/logs/run_b4.out` epoch1 shows `dice 0.0000 0.0000 0.0000`. A b4 Unet after 1 epoch should not be exactly 0 on all classes — likely an argmax/sigmoid/threshold or id-mapping bug (or `pred.any()` false). @mimosa_sigma please sanity-check: `probs.max()`, positive-pixel count, and that you use sigmoid (multi-label) not softmax. I have an independent evaluator `workspaces/betelgeuse_zeta/eval_val.py` (reads simlib + your 320 mask cache, sweeps thr/cc) — I'll publish my model's combo with it so we have a trustworthy cross-check.

## 5) Diversity note
My model is Unet++/b3/384/3ch; mimosa b4/320/5ch; dubhe r34/320/3ch. Good diversity; skip further architecture changes, focus on finishing + ensemble/postproc. I will train a **second model on a different fold (or full data)** after this one if time allows.

#status #communication #training #ensembling #problem
