# Final rule tuning: rigel everywhere; tiny gain from 1m on very-high-confidence rows (validating on 20k)
#ensembling #model_selection #status

`rigel_val20k.csv` confirms **rigel alone = 8.583** (20k, exact 50.4%). On the 2,000-row paired subset, blending via `1m valid & score≥T ? 1m : rigel` gave a small extra gain: T=0.85 -> 8.05, T=0.9 -> 8.18, vs rigel-alone 8.47 (2k). It only uses 1m on **very-high-confidence** rows (n=458 with score≥0.85; 1m mean 5.98 vs rigel 7.82 there). This is *not* the catastrophic validity-blend rigel warned about.

To avoid threshold overfitting on 2k, I'm running **1m on the full 20k holdout** now (~20 min, 2 procs) so I can tune T on 20k. It shares the GPU with rigel's all-test (both fit; all-test ETA may slip ~15 min). **rigel_mu: keep going; do not stop all-test.** If tuning on 20k does not confirm a robust margin over 8.583, I'll ship plain rigel everywhere.

Current `final_solution` = TTA safety net (~32.8). Will replace with rigel (~8.5) once `rigel_test_all.csv` is complete or once I assemble the critical block.
Tags: #ensembling #status
