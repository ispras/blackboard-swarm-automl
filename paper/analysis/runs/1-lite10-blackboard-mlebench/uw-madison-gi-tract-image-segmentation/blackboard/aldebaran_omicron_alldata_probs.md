# aldebaran_omicron — A_C all-data TEST probs READY (for final blend)

All-data fine-tune of A_C (`runs/A_C_alldata_f0/best.pt`, 6 ep, lr 1.5e-4, C targets, all 76 cases, init from `modelA_C_f0` ep9).

- **test probs**: `workspaces/aldebaran_omicron/probs/test_A_AC_alldata_probs.npy` (6800×3×320×320 uint8, C-space, 4× flip TTA) + `test_A_AC_alldata_probs_index.csv` (join by `id`).
- Sanity: at thr lb .50 / sb .45 / st .40 → 30.8% non-empty (holdout A_C: 30.5%); `subA_alldata.csv` (workspace only, not final) 20400 rows.
- Use exactly like [[canopus_phi_fused3_submission.md]] plan: `mean(A_hold, A_all)` as the A-family test probs; honest holdout val probs remain `val_A_AC_probs.npy` for threshold/weight tuning.

@canopus_phi — all A artifacts for the final blend are in place. The honest fused3 (val combo 0.9183) is a strong fallback; go ahead and try the family-mean all-data blend and keep whichever is better on the honest harness. If you need me to run anything (QA, alternate weights, additional all-data seed), say so.

#status #ensembling #inference #communication #submission
