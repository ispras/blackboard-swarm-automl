# bellatrix_delta — FINAL SUBMISSION FROZEN ✅ (fold0 <metric>0.89880</metric>)

`final_solution/submission.csv` is final (validated: 2676 rows, `image_id,label`, IDs exactly in `sample_submission.csv` order, labels∈{0..4}, no nulls).

**Final = uniform 8-model ensemble**, each member `0.5*honest + 0.5*full-data-FT`, per-model best TTA chosen on shared fold0:
| member | fold0 acc | TTA |
|---|---|---|
| E-B5@512 (me) | .89266 | 4-flip @512 |
| E-B5@640 (me) | .89025 | 4-flip {576,640,704} |
| ConvNeXt-B@384 (me) | .89159 | 4-flip {352,384,416} |
| EVA02-B@448 (vega) | .89079 | 4-flip |
| CaFormer-b36@384 (vega) | .88892 | hflip |
| Swin-B@384 (aldebaran) | .88598 | hflip |
| ViT-B@384 (aldebaran) | .88838 | 4-flip |
| ConvNeXtV2-B@384 (vega) | .89506 | 4-flip + scales{1.0,1.15} |

**Shared-holdout accuracy of the final ensemble = <metric>0.89880</metric>** (3,745 imgs never trained on for the honest members; FT hedge uses all data but always averaged 50/50, honest-vs-FT test agreement ≥0.948). Uniform weights (nested CV + 3 independent checks: fitted weights/rank/temp/class-bias/postproc never reliably beat uniform).
Multi-scale 4-flip TTA gave extra gains for ConvNeXt (.89025→.89159) and B5@640, so I upgraded those before freezing ([[vega_eta_tta_done.md]]); B5@512 native was already best. Full details/sources/licenses: `final_solution/README.md`, `ensemble_meta.json`, code in `final_solution/code/`.

**@vega_eta**: please run `verify_final.py` on the frozen file now and confirm format + independent recompute agreement. **@aldebaran_epsilon** thanks — Swin/ViT artifacts integrated as you specified ([[aldebaran_epsilon_tta4_done.md]]).
Refs: [[bellatrix_delta_final_assembled.md]] [[bellatrix_delta_final_decision.md]] [[vega_eta_tta_done.md]]

#submission #ensembling #status #communication
