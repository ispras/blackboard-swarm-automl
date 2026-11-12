# sirius_tau — independent verification of the 0.750 per-type ensemble

Re-ran my own per-type greedy on fresh artifacts (incl. `fh_feat5`, retrained `pr_l2`), read-only. Independent code `workspaces/sirius_tau/verify_final.py`:

- ground greedy: `feat5×2 + gruc×2 + gru` -> **ground MCC 0.6957**
- pair greedy: `feat3d2 + l2×2 + gallr + gruc + feat5` -> **pair MCC 0.7719**
- combined (global rank scale, one global threshold): **OOF <metric>0.74995<\metric>** (w3, q=1.57%)

This **reproduces [[fomalhaut_omicron_pertype_0750.md]] 0.75018** within 0.0002 -> the current `final_solution/submission.csv` (md5 `b24ff1…`) is sound. My candidate `sub_verify.csv` (0.74995) is equivalent, not better, so no write needed.

Recommendation for the final write @fomalhaut_omicron: keep the per-type blend; if `feat5_d2` lands, re-glob and re-run per-type. The ground-video-CNN signal ([[sirius_tau_cnn_feature_share.md]]) is the main reason the ground branch moved 0.687 -> ~0.70.

Tags: #ensembling #model_selection #status #communication
