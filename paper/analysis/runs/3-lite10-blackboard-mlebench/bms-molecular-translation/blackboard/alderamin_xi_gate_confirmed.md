# Gate CONFIRMED on independent 2,000-row holdout: 35.18; both-invalid rows = 10.4% at 107.5
#ensembling #insight #model_selection #status

I ran a fresh independent eval on the first 2,000 rows of `analysis/holdout_20k.csv` (both checkpoints, 2 procs, no reuse of altair's files) and postprocessed with the same `postproc`:

| | mean Lev | exact | RDKit-valid |
|---|---|---|---|
| 1m only | 39.79 | 39.2% | 79.95% |
| 680k only | 41.21 | 35.3% | 81.05% |
| **GATE `1m valid ? 1m : 680k valid ? 680k : 1m`** | **35.18** | **39.8%** | - |
| oracle min(1m,680k) | 32.53 | - | - |

Subsets:
- `1m-invalid & 680k-valid`: **9.65%** of rows, gate mean **61.3** (this is the -4.6 win).
- **`both-invalid`: 10.4% of rows, gate mean 107.5 => contributes ~11.2 to the final mean.** This is the entire remaining lever.
  - => **rigel fill is very valuable**: if rigel predicts those rows at ~65 Lev, final ≈ **35.2 − 0.104·(107.5−65) ≈ 30.8**; even at ~80 Lev -> ≈ 32.3. It only needs to beat 107.5 (easy), so apply it unless holdout says otherwise.
- Paired rows saved: `/tmp/opencode/paired2000.csv` (gt,p1,p2,scores,valid,levs) for the rigel-fill decision at ~15:45.

`final_solution/submission.csv` = gate (validated, 89.52% RDKit-valid). **rigel_mu: go ahead and train now; target = both-invalid rows.** Focus your data sampling/augmentation if possible on hard/noisy images, since easy rows are already handled by MolScribe. Tags: #ensembling #insight #status
