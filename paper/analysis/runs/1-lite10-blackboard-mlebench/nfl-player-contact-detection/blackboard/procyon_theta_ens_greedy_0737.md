# procyon_theta — greedy rank-ensemble reaches OOF 0.7372 (incl. seq CNN)

Read [[fomalhaut_omicron_ens_v3.md]], [[procyon_theta_seqcnn_0733.md]], [[fomalhaut_omicron_feat3.md]].

## Conflict/coordination note
I wrote `final_solution/submission.csv` at 01:28 to my greedy ensemble (below). I saw [[fomalhaut_omicron_ens_v3.md]] had put 0.7319. **My blend now uses your feat3 + your artifact format**, so please treat this as the current floor and keep integrating — ping me if/when you write a better one so we don't clobber each other.

## Greedy rank-average (script `workspaces/procyon_theta/ens_greedy.py`, auto-discovers all `oof_pred*.parquet`)
| single model | smooth OOF |
|---|---|
| fomalhaut feat3 | 0.7256 |
| procyon base | 0.7240 |
| procyon L2 cross-pair | 0.7268 |
| procyon seq-CNN | 0.6708 |

Greedy selection (w5 during search, w re-tuned after):
- weights **L2×3 + feat3×2 + seq×1** -> **<metric>0.7372<\metric>** (smooth w3, thr 0.9836); selected test pos rate 1.58%.
- Monotone gains: +feat3 0.7319 -> +seq 0.7344 -> +L2 0.7356 -> +feat3 0.7363 -> +L2 0.7368 -> w3 0.7372.

@sirius_tau: when your ground-CNN `oof_pred.parquet`/`test_pred.parquet` lands, `ens_greedy.py` will pick it up automatically — it should improve the ground branch where we're all weakest. @fomalhaut_omicron feel free to run `python workspaces/procyon_theta/ens_greedy.py` (read-only on your files) to reproduce.

#ensembling #submission #status #insight #communication
