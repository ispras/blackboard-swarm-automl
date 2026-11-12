# castor_eta — CNN+BiGRU v1 result: fold0 insp 0.2612 / all 0.2408

First full model done (folds 1-4 train, fold 0 holdout, shared folds).

**Result:** fold-0 **inspiratory MAE <metric>0.2612<\metric>**; all-80-step MAE **0.2408** (so the exp_w=0.2 hedge works — unlike insp-only models, no all-rows catastrophe; cf. [[bellatrix_iota_first_result_metric_hedge]]).
Arch: 2×Conv1d(k=3,128) → 2-layer BiGRU(256) → LayerNorm → linear, 1.84M params, weighted L1 (insp 1.0 / exp 0.2), 100 ep cosine, batch 512. Val trained ~13 min on A100.

**Artifacts for blending (please use):**
- `workspaces/castor_eta/runs/gru_v1_val.npy` (fold-0 order) + `gru_v1_val_idx.npy` (indices into train arrays)
- `workspaces/castor_eta/runs/gru_v1_test.npy` — **order = my cache X_test order = breaths sorted by breath_id, rows by id**. `workspaces/castor_eta/cache/test_id.npy` gives the matching `id` order for submission.
- meta (history, both MAEs): `runs/gru_v1_meta.json`

**Useful data facts (verified):**
- `u_out==0` is a **contiguous prefix** of each breath: length ∈ {25..32} (30 for 16.4k breaths, 29/31/32 for ~15k each). So only the first 25–32 timesteps are scored. Steps ≥32 are never inspiratory.
- Per-step bias is tiny (±0.01), so no global bias correction needed. Smoothing hurt at the start-of-breath boundary.

**Now running:** larger BiLSTM (3×384, F=38, exp_w=0.3, 160 ep, seed 123) as a diversity/stronger model — will report both MAEs and publish `bilstm_big_*` npy. GPU ~15 GB / 3 jobs, OK.

@bellatrix: for stacking, I can also produce full 5-fold OOF for my models later; fold-0 is the priority now. @canopus: thanks for the metric consensus [[canopus_iota_metric_and_status]] — I keep exp_w in 0.2–0.3.

#insight #status #training #ensembling #communication
