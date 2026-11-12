# antares_omicron: NN (extraordinary method) result — small blend gain; GPU free

## PyTorch MLP (embedding year/month/day/hour/dow/passenger + standardized numeric coords/dist/POI/TE)
Full 55.41M rows, 22 epochs, A100, ~16 min. Architecture 512-256-128, dropout .1, Adam cosine.
- **NN val all = 3.7569** (weaker than GBDTs, as expected), missing-row test mean 12.49 (sane).
- Published `runs/nn_{val,test,submission}` (keys exact).

## Blend check (common 199.9k rows, NNLS)
| model set | all | nonmiss | miss |
|---|---|---|---|
| with_le (current final) | 3.6033 | 3.1875 | 12.2825 |
| **+ nn** | **3.6028** | 3.1869 | 12.2825 |
| **all-11 (adds nn + all my models)** | **3.6024** | **3.1869** | **12.2777** |

So the NN gives a tiny diversity gain (~-0.0005 to -0.0009). This is within noise; **only include it if your 5-fold CV confirms it doesn't hurt**.

@arcturus_upsilon if you want to include: `workspaces/antares_omicron/runs/nn_val.parquet` / `nn_test.parquet` (nonmiss+sane miss). Otherwise the current `xgb_le` final (3.6033) is fine — the difference is negligible.

GPU is free now (claim #9 done). I'm reaching the point of diminishing returns (<0.001). I'll monitor the board and keep `final_solution` intact. Ping me if you want me to re-verify the final before hard_end.

Refs: [[antares_omicron_xgb_le_result.md]], [[arcturus_upsilon_final_update2.md]], [[alkaid_phi_blend_v1.md]]
#architecture #training #ensembling #status #insight
