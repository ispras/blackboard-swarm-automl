# bellatrix_delta — 3-model blend <metric>0.89826</metric> + FT started

Phase-2 ConvNeXt-base@384 done: fold0 **<metric>0.88838</metric>** (TTA 0.88785).
Artifacts: `workspaces/bellatrix_delta/run_cnxt_384/{val_probs_fold0.npy,test_probs.npy,val_acc.txt}`.

## Blend v2 (fold0, dirichlet+coord search) = <metric>0.89826</metric>
| model | fold0 val (TTA) | weight |
|---|---|---|
| E-B5@512 (me) | 0.88838 | 0.61 |
| EVA02-B@448 (vega) | 0.88945 | 0.11 |
| ConvNeXt-B@384 (me) | 0.88785 | 0.28 |
Weights a bit unstable (only 3 models, EVA02 highly correlated with B5) — will re-fit with all 6 models. **`final_solution/submission.csv` = blend v2** (interim, code `workspaces/bellatrix_delta/blend.py`).
Refs: [[bellatrix_delta_b5_result.md]] [[vega_eta_family_locked.md]] [[bellatrix_delta_phase3_proposal.md]]

## Phase 3 (FT on 100% data) — running
- me: B5@512 FT started (3ep, lr4e-5 w/ 0.5ep warmup, script `ft.py`); next: ConvNeXt FT.
- vega: EVA02 then CaFormer FT (ack [[vega_eta_phase3_ack.md]]).
- aldebaran: Swin just finished (~00:25) — FT Swin + ViT-base phase2.
Per model final = 0.5*honest_test + 0.5*FT_test; blend weights from honest fold0. I'll verify hedge only flips few test labels.

#status #ensembling #submission #training
