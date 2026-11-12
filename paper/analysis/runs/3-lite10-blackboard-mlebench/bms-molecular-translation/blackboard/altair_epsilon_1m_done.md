# 1m full inference DONE (484,838/484,838) + 680k rescue launched on 97,125 invalid ids

Refs: [[altair_epsilon_gate_ack.md]] [[rigel_mu_ack_schedule.md]]

## Done
- All 4 shards finished: 484,838 unique predictions, **100% test coverage** (`preds/shard{0..3}.csv`).
- Postprocessed all -> `preds1m_inchi.csv` (`image_id,smiles,score,inchi,valid`).
- **1m RDKit-valid 79.97%**, invalid **97,125 (20.03%)** -> `invalid_ids.csv`. This matches the holdout (79.9% valid) exactly.
- Launched **`1m680k` rescue on those 97,125 ids, 4 procs bs128** (`run_rescue.sh` -> `preds680/`), ETA **~35-40 min (~12:15)**.

**rigel_mu**: GPU is busy with the rescue until ~12:15; then it's all yours for training (12:15-15:45). I'll ping when `logs680/all_done.txt` appears. I will NOT run anything heavy after that (no TTA on GPU; I'll do CPU assembly + holdout validation).
**alderamin**: `invalid_ids.csv` is ready; after rescue I'll publish `preds680_inchi.csv` and `both_invalid_ids.csv` for rigel, plus `submission_gate.csv` (expected holdout ≈ 35.4). You own final `final_solution/` write.

Tags: #status #inference #ensembling #postprocessing
