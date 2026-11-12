# TTA rescue on both-invalid rows found a real gain; running it now (~35 min), then GPU to rigel

Refs: [[altair_epsilon_rescue_done.md]] [[rigel_mu_ack_schedule.md]]

## Holdout experiment on the 53 `both-invalid` rows (1m rot0 and 680k rot0 both RDKit-invalid)
Current gate output on those rows = formula-only string, mean Lev **99.2**. Rotating the image and re-decoding:
| selection | valid | mean Lev on these 53 |
|---|---|---|
| 1m rot90 + 680k rot180 (first valid) | 50.9% | **88.5** |
| 1m rot90 + 680k rot90 + 680k rot180 | 60.4% | 86.1 |
| all 6 (1m/680k x 90/180/270) | 69.8% | 81.5 |

=> on the ~10.5% both-invalid rows this is a **guaranteed ~ -1.1 to -1.9 overall** (no extra model needed, no risk to good rows).

## Running now
`run_tta.sh` (4 procs, bs128): `1m` rot90 + `680k` rot180 on the **50,694 both-invalid test ids** -> `preds_tta/`, ETA **~30-35 min (~12:55)**. Then I merge with order `1m_r90 -> 680_r180 -> keep formula fallback`, produce `submission_gate_tta.csv`.

**rigel_mu: GPU fully free at ~12:55** (I'll ping with `logs_tta/all_done.txt`); please start training then, 12:55-15:45. For the final rule we'll use `1m-valid ? 1m : (680k-valid ? 680k : (tta-valid ? tta : (rigel-valid ? rigel : formula)))`.
**alderamin**: holdout numbers above; please independently verify the TTA gain (files `/tmp/tta_both_preds.csv`, `/tmp/tta_both_base.csv`) and keep ownership of final_solution write.

Tags: #augmentation #ensembling #inference #insight #plan #status
