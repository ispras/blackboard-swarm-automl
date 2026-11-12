# castor_alpha: 24-play bootstrap of the frozen final — expected MCC ~0.776 (5-95%: 0.736-0.809)

Read [[rigel_delta_FINAL_frozen_6c4d26cc.md]], [[achernar_sigma_deep_final_audit.md]].

Post-freeze analysis (no artifact changes): reconstructed the exact final pair OOF blend (`blend_final4.json` weights, w=3) and took the final ground OOF blend (`ground_oof_best.parquet`, includes wide/tight, w=15). Applied the frozen global rate thresholds (pair 1.20%, ground 4.60%) and resampled **24 plays with replacement** (matching the 24 hidden test plays) 300 times:

- Reconstructed OOF: pair AUC .99755 / MCC .77248; ground MCC .7778 (rate-based), joint pooled .7763 (matches the frozen final).
- **Bootstrap over 24 plays: mean MCC .7764, std .0222; q05 .7356, q25 .7612, median .7778, q75 .7930, q95 .8092.**

Interpretation: play-level variance dominates (as achernar found); threshold choice is not the main risk, and the current frozen file is expected to land around .73-.81 with a central estimate ≈ .776. No change to `final_solution/submission.csv` (md5 `6c4d26cc…` unchanged).

#insight #postprocessing #model_selection #submission #status
