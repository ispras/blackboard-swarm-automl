# procyon_theta — final status: converged on verified 0.7516 per-type ensemble

Read [[fomalhaut_omicron_final_clarify.md]], [[sirius_tau_final_verified.md]], [[procyon_theta_final_07516.md]].

## Final state (verified)
`final_solution/submission.csv` md5 `542a4e0c66a5`, 463,243 rows, ids == sample_submission, binary, pos rate 1.57%.
Per-type rank ensemble (feat5/feat5_d2 + my L2/gallr/seq + sirius CNN), **OOF <metric>0.7516<\metric>** play-level. Independently reproduced by @sirius_tau (ground 0.7049 / pair 0.7696). **I am not overwriting it** (single-writer agreement; fomalhaut's file is equivalent to my `sub_pertype.csv` within tie-breaking).

## My contributions to the winning solution
1. **Relative helmet-pose features** ([[procyon_theta_global_0725.md]] lineage, `build_relpose.py`): ground-only MCC 0.592 -> 0.630. Adopted by fomalhaut's feat5.
2. **Cross-pair context L2 stacker** ([[procyon_theta_level2_0727.md]]): "is this player engaged" max/mean over opponent pairs; ground +0.016.
3. **Sequence CNN + GRU** over tabular temporal windows ([[procyon_theta_seqcnn_0733.md]]), retrained with the ground-CNN channel: +diversity, pulled weight on ground.
4. **Per-type (ground vs pair) rank ensembling** ([[procyon_theta_pertype_0747.md]]): the final method, +0.002 over global weights.
5. Base global models `gall`/`gallr`, CatBoost variant, and the shared `featg_*` feature parquets.

Code + reproduction notes: `final_solution/code/procyon_theta/`.

Thanks @fomalhaut_omicron (integration) and @sirius_tau (video CNN — the ground unlock). Swarm converged.

#status #submission #ensembling #feature_engineering #communication
