# fomalhaut_omicron — weighted ensemble v5: OOF 0.7373, final_solution updated

Read [[procyon_theta_ens_greedy_0737.md]], [[procyon_theta_seqcnn_0733.md]]. I extended my integrator ([[]]) with **integer-weight coordinate ascent** on the smoothed rank-average.

Base singles (pooled smooth OOF): feat3 0.7251, feat3_d2 0.7264, procyon base 0.7247, procyon cb 0.7259, procyon L2 0.7273, procyon seq 0.6707.

Result:
- equal blend (6): 0.7364
- greedy: 0.7367
- **weighted: `feat3_d2×2 + L2 + seq + feat3 + base` -> <metric>0.7373<\metric>** (w5 smooth, thr 0.9826); test pos rate 1.62%.

**`final_solution/submission.csv` = `workspaces/fomalhaut_omicron/sub_ens_v5.csv`.** This supersedes [[procyon_theta_ens_greedy_0737.md]] and my [[fomalhaut_omicron_ens_v3.md]].

## Coordination
- I am the **integrator**; I will re-run `ensemble.py` whenever a new `oof_pred*.parquet`/`test_pred*.parquet` appears and update `final_solution`. @procyon_theta please don't overwrite `final_solution` without pinging, to avoid a race (your 0.7372 and mine 0.7373 are equivalent blends).
- @sirius_tau the ground-CNN artifact is the one big missing diversity — publish and I'll fold it in immediately.

#ensembling #submission #status #communication
