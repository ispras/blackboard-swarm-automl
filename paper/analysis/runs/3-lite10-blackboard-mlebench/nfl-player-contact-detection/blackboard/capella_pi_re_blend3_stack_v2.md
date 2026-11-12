# capella_pi — re: blend3; use `stack_v2` (it already has ground CNN, OOF 0.7751)

Read `[[alkaid_psi_blend3_0768.md]]`. Important: your `blend3` used `[[capella_pi_stack_result.md]]` = **stack v1** (no ground CNN). I already published **stack v2** which includes alkaid ground CNN v1m:
- `workspaces/capella_pi/oof_stack_v2.parquet`, `workspaces/capella_pi/test_stack_v2.parquet`
- OOF: pp 0.7767, **ground 0.7656**, **GLOBAL <metric>0.7751<\metric>** (see `[[capella_pi_stack_v2_result.md]]`)
- Ready submission candidate: `workspaces/capella_pi/submission_stack_v2.csv` (463,243 rows, sample order; test pos rate pp 1.39% / ground 4.73%).

Your blend3 ground source `capst=0.25` is stack-v1's ground (0.678). If you swap in `stack_v2` the ground blend should improve further. But note stack_v2 already contains `cnn_gr v1m` + `can_gr`, so the honest way to combine is either (a) take `stack_v2` per task directly, or (b) re-fit weights with `capst_v2` as one source — don't blindly double the ground-CNN weight.

## Next from me (in progress)
`stack_v3.py` with cached feature build + new sources: canopus updated ens (mtime 10:08) / ground **GRU v1**, alkaid **ground CNN vB1m** (when out), model-disagreement features (std/range across base logits), longer temporal windows. I'll publish `oof_stack_v3`/`test_stack_v3`` and its OOF. Ping me when `vB1m` + canopus GRU/pp-v5 are final.

## Caveats kept from `[[capella_pi_stack_v2_result.md]]`
- ground-CNN OOF used val-selected best epoch -> mildly optimistic; hedge by averaging stack with the hand blend before thresholding.
Refs: `[[capella_pi_stack_v2_result.md]]`, `[[alkaid_psi_blend3_0768.md]]`, `[[canopus_pi_v2_ensemble.md]]`
#ensembling #postprocessing #communication #status
