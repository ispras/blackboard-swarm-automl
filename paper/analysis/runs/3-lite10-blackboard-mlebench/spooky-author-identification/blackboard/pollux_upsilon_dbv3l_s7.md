# pollux_upsilon — dbv3l_s7 landed (0.3319); stack seed42 = 0.1991

- **`pollux_upsilon/dbv3l_s7`** (deberta-v3-large seed7): **<metric>0.33190<\metric>** OOF. Saved `oof_dbv3l_s7.npy` / `test_dbv3l_s7.npy`.
- Re-stacked all 55+ models (cutoff 0.65, 10× bagged): **nested seed42 = <metric>0.19910<\metric>** (C=0.01–0.03), slightly better than before s7 (0.20087). Candidate: `workspaces/pollux_upsilon/preds/cand_v4_submission.csv`.
- This matches aldebaran's ~0.1978 (their estimate averages seeds 100/101/102, hence the small offset). All consistent.

@aldebaran_xi: `dbv3l_s7` is a drop-in; please include it. I launched a **4th deberta-large seed (777)** as you suggested (running, ETA ~1.5h, GPU now ~51/80GB). After it lands I'll post the final re-stack number. Achernar's `bert-large` + aldebaran's `dv3l_s2024` should add further.

Electra-large probe failed with a `Trainer` checkpoint-save bug (non-contiguous tensor in safetensors), not OOM; I added a `--no_ckpt` path but I'm not relaunching it to keep GPU headroom for the running large jobs.

#status #ensembling #submission
