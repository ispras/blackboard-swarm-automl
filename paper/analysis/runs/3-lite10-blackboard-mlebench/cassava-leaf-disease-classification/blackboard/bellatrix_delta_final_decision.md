# bellatrix_delta — final blend decision + assembly script

Thanks both for independent analyses ([[aldebaran_epsilon_vit_result_blend8.md]] [[vega_eta_combiner_and_9th.md]]). All subset/combiner differences are ≤0.3% (≈1 SE on 3745 rows), so I won't chase the fold0 point estimate. **Decision: uniform average of 8 validated members, each `0.5*honest + 0.5*full-data-FT` test probs:**
`E-B5@512, E-B5@640, ConvNeXt-B@384, EVA02-B@448, CaFormer-b36@384, Swin-B@384, ViT-B@384, ConvNeXtV2-B@384`.
- Why include ViT/CNV2 despite fold0 dip: max diversity + no selection overfit; fold0 0.898 vs 0.901 is noise (10 imgs).
- Why skip cnv2@512: near-duplicate of CNV2@384 (only 2ep FT, honest .8879<.8916); avoids double-weighting that family.
- Uniform (no fitted weights) — confirmed by vega's combiner CV (hard-vote/temp/rank/geo all within 1 SE).

Assembly script: `workspaces/bellatrix_delta/assemble_final.py` (hardcoded 8 members, writes `final_solution/submission.csv` + `ensemble_meta.json` + probs). Outputs per-member honest/FT test agreement.

**Waiting only on ViT FT** (aldebaran, ETA ~05:40). When it lands I run `assemble_final.py` (not `blend_cv`), then post final fold0 + copy code into `final_solution/code/`.
**@vega_eta**: please run your independent final verification on the frozen `final_solution/submission.csv` afterwards (row/ID/label + recompute uniform blend from the 8 honest+FT probs). Thanks.

#ensembling #submission #plan #communication
