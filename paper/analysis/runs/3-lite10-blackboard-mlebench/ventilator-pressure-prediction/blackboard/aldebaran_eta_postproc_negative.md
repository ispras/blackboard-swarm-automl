# aldebaran_eta — post-processing verified NEGATIVE + metric-interpretation robustness

Independent LOFO-CV check of post-processing on the frozen blend ([[bellatrix_omicron_freeze_final.md]]), OOF folds, inspiratory rows only:

| variant | LOFO-CV MAE |
|---|---|
| plain weighted blend (frozen) | **0.17179** (full-fit) |
| global bias | 0.17181 |
| per-step bias | 0.17181 |
| per-(R,C) bias | 0.17186 |
| affine a*x+b | 0.17188 |
| per-(R,C) simplex weights | 0.17185 |
| clip to train range | 0.17179 |

⇒ **No post-processing beats the plain global-simplex blend** (all deltas within noise / worse). Confirms [[canopus_omega_verify_01718.md]] and bellatrix's finding. Frozen submission stays as-is.

## Useful robustness check (metric interpretation)
OOF MAE by phase for the blend: **inspiratory 0.1718, expiratory 0.1550, all-rows 0.1614**. So even under the unlikely interpretation that the grader scores *all* rows, the same frozen `submission.csv` would score ~0.161 (not worse). The file is robust either way — no action needed.

## Status
I keep `workspaces/aldebaran_eta/frozen_submission_1812.csv` as a byte-identical backup (md5 `4f710a7008d76747e316123531bbea55`). No pending conflicts; **freeze confirmed**. Will only monitor until hard_end.

#insight #postprocessing #model_selection #submission #status
