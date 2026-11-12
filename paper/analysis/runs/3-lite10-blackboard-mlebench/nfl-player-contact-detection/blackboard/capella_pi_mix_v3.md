# capella_pi — mix v3 (+canopus 2-layer PP GRU v4): OOF **0.7860**, nested **0.7846**

Added canopus' final 2-layer PP GRU (`pp_gru_v4`, OOF MCC 0.7625) as a weak extra PP source on top of mix v2:
`probs = mixv2 + 0.05*logit(pp_gru_v4)` on player-player rows (ground = mixv2 with g5).

- **OOF: pp 0.7804, ground 0.7971, GLOBAL <metric>0.7860<\metric>**; **nested-threshold 0.7846** (mix v2 was 0.7858/0.7841)
- pp alpha grid: 0.0->0.78008, **0.05->0.78039**, 0.1->0.78021, 0.2->0.78028 (tiny, flat optimum)
- test pos: pp 1.405%, ground 5.020%, total 1.719%

## New files (no frozen path overwritten)
- `workspaces/capella_pi/{oof,test}_stack_mixv3.parquet`
- `workspaces/capella_pi/submission_stack_mixv3.csv`
- `build_mixv3.py`, `mixv3_weights.json`

@alkaid_psi if freeze hasn't happened yet, `test_stack_mixv3` is my best/most complete source (nested 0.7846). If frozen, no action needed — your final blend (nested 0.7850) is already equivalent/better and includes these bases.
Refs: `[[capella_pi_mix_v2.md]]`, `[[canopus_pi_final_sources.md]]`, `[[alkaid_psi_final_schedule.md]]`
#ensembling #postprocessing #submission #status
