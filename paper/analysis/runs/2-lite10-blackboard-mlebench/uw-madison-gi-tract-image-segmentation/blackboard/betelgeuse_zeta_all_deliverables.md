# betelgeuse_zeta — ALL deliverables final (OOF b4 0.9484) + b4-all test probs

Read [[betelgeuse_zeta_b4_probs_ready.md]], [[mimosa_sigma_ensemble_ready.md]], [[betelgeuse_zeta_fold0_oof_combo.md]].

b4/384 fold0 OOF combo (my fast evaluator, same protocol as before):
<metric>0.9484</metric> (dice .9294) — lb 0.9385 (thr .60, cc3d 300), sb 0.9337 (thr .50, cc3d 300), st 0.9730 (thr .55, cc3d 300).

## Complete deliverable set (all 320px float16, hflip-TTA)
| role | file in `workspaces/betelgeuse_zeta/` | OOF combo |
|---|---|---|
| Unet++/b3/384 fold0 OOF | `betel_f0_valprobs.npy` + `betel_f0_valid.npy` | **0.9501** |
| Unet++/b3/384 fold0 test | `betel_f0_testprobs.npy` | — |
| Unet++/b3/384 all-data test | `betel_all_testprobs.npy` | (use f0 OOF) |
| Unet/b4/384 fold0 OOF | `betel_b4f0_valprobs.npy` + `betel_b4f0_valid.npy` | **0.9484** |
| Unet/b4/384 fold0 test | `betel_b4f0_testprobs.npy` | — |
| Unet/b4/384 all-data test | `betel_b4all_testprobs.npy` | (use b4f0 OOF) |

All `*_testprobs.npy` are `[6800,3,320,320]` in **`cache/test_ids.json` order** (ids in `betel_*_test_ids.npy`); val probs are in fold0 order with matching `_valid.npy` ids.

@mimosa_sigma recommended use: OOF members = `betel_f0` (0.9501) and `betel_b4f0` (0.9484); test counterparts = `betel_all` and `betel_b4all`. You now have up to 5 OOF members (your b4v2, dubhe b3-Unet, dubhe FPN if ready, my b3-Unet++, my b4-Unet). Keep only what improves the honest holdout. My two models are fairly correlated (both b3/b4 Unet-ish, 384, 3ch, same folds) — cap their combined weight if the weight search prefers.

I am **done training** (~1 slot freed). Backup standalone submission from my `betel_all` probs (thr .6/.6/.6, cc2d 200) is at `workspaces/betelgeuse_zeta/submission_betel_all.csv` — validated: 20400 rows, exact `data/test.csv` id/class order, all RLE pairs sorted/positive/in-bounds, 5598 non-empty. If the ensemble submission is ever missing/corrupt near the freeze, this is a working fallback. @dubhe_zeta ping me if you want an independent RLE/format verification of the final file before you freeze.
#status #ensembling #submission #communication #model_selection
