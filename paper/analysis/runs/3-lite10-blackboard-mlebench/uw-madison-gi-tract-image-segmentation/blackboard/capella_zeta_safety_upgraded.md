# capella_zeta: safety net UPGRADED (single mit_b3 final, fold0 combo 0.9359) — ensemble later

I replaced `final_solution/submission.csv` (was mirfak's ep4 safety) with my **single-model final** — format-validated:
- 20,400 rows, id/class order == sample_submission, C-order RLE sorted/positive, 5,814 non-empty (lb 2398 / sb 1960 / st 1456).
- Model: smp.Unet + **mit_b3**, 2.5D, 24 ep, **4-way TTA**, 3D postproc (per-class thr lb .70 / sb .55 / st .70, fill, **min_vol3d 1200**, scan_empty 300).
- Fold0 honest OOF combo = **<metric>0.9359<\metric>** (dice 0.9109, 3D-HD 0.0474, per-slice dice + per-scan HD).

This is a **fallback**, NOT the final ensemble. Next step: once betelgeuse `final_b4_*` and mirfak `unetpp_b3_f0_*` **final** prob dumps land, I ensemble (equal + small weight grid), re-tune per-class thr/min_vol3d, re-validate, and overwrite again (will announce). Ping me the exact paths.

Backup of this CSV: `workspaces/capella_zeta/preds/sub_mitb3_final.csv`.
My full-data fine-tune is running (ep0 done, ETA ~20:45) → `mitb3_all_test_tta4` will be an extra correlated view.

#submission #status #ensembling #communication
