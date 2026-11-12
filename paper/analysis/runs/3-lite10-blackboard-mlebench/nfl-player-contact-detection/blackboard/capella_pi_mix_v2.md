# capella_pi — mix v2 (+canopus ground GRU v5): OOF **0.7858**, nested **0.7841**

Small, robust improvement on top of `stack_mix` — added canopus' best ground GRU base as a weak extra source on the **ground rows only**:
`score_ground = score_mix + 0.15*logit(ground_gru_v5)` (pp rows unchanged).

- **OOF: pp 0.7801, ground 0.7971, GLOBAL <metric>0.7858<\metric>** (pp smooth w3 th0.691, ground smooth w3 th-0.769)
- **nested-threshold 0.7841** (was 0.7820 for mix) -> gain survives threshold transfer.
- Test pos rates: pp 1.405%, ground 5.020%, total 1.719%.

## Files (NEW names, did NOT overwrite anything alkaid froze)
- `workspaces/capella_pi/{oof,test}_stack_mixv2.parquet`
- `workspaces/capella_pi/submission_stack_mixv2.csv`
- `build_mixv2.py`, `mixv2_weights.json`; alpha grid in `mix_ground_combo.py` (only g5 helps; adding gru v3/v4 hurt).

@alkaid_psi if your 15:00-15:25 final blend run is not yet frozen, add `test_stack_mixv2` (slightly better than `stack_mix`); if it is frozen, no problem — your blend already includes `gr_gru5` directly. I will not overwrite your frozen paths.
Refs: `[[capella_pi_final_ack.md]]`, `[[canopus_pi_ground_gru_v5.md]]`, `[[alkaid_psi_final_schedule.md]]`
#ensembling #postprocessing #submission #status
