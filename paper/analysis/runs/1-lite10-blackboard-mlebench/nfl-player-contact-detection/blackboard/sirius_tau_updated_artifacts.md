# sirius_tau — updated artifacts (2-seed ground CNN), OOF 0.7068

Improved the video branch by **averaging 2 CNN seeds** (v1+v2, logits) -> ground CNN OOF AUC 0.8835 -> **0.8854**. Updated files in place:
- `workspaces/sirius_tau/oof_pred.parquet` / `test_pred.parquet` — smoothed **OOF <metric>0.7068<\metric>** (was 0.7053), ground AUC ~0.955+, pair ~0.9974.
- `workspaces/sirius_tau/ground_cnn_feat_{train,test}.parquet` — updated to the 2-seed CNN logit (same schema; reuse in [[procyon_theta_ens_0745_coord.md]] L2 / [[fomalhaut_omicron_feat3.md]] feat if you have time for one more retrain).
- `sub_final.csv` updated (pos rate 1.47%).

@sirius_tau-cnn already credited in [[procyon_theta_ens_0744.md]] (CNN was the ground unlock) — this is a small extra. @fomalhaut_omicron @procyon_theta: please keep **one writer** on `final_solution`; current best per board 0.7446. I'll stop touching it unless asked.

Tags: #ensembling #submission #status #insight
