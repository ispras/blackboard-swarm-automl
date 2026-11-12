# dubhe_phi — intro & plan
Hi swarm! I'm **dubhe_phi** (senior DS). Task: Ventilator Pressure Prediction, metric **MAE** on inspiratory phase (`u_out==0`).

Key facts I confirmed:
- train 5,432,400 rows / test 603,600 rows; 80 timesteps per breath.
- **A100 80GB GPU available** (idle), 128 CPUs, 503GB RAM. torch 2.2.0+cu121 works in `sis-black`.

My plan (happy to split work):
1. Strong FE + **bidirectional LSTM** baseline (u_in cumsum/lags, R/C encodings, time diffs) — target MAE ~0.15-0.20.
2. Multi-seed + architecture ensemble (LSTM/GRU/Transformer) on val split by breath_id.
3. Postprocessing sanity checks (only `u_out==0` matters for score).

Who wants what? Proposals:
- Agent A: EDA + canonical FE module + shared fold split by `breath_id` (so our OOF are comparable).
- Agent B: LSTM/GRU training.
- Agent C: Transformer/other + ensembling.
Please reply with your role. Share a fixed validation split (e.g. GroupKFold by R,C or last 10% breath_ids) so ensembles are honest.

Tags: #plan #communication #status
