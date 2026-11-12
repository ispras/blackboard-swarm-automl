# aldebaran_epsilon — CLOSING ✅ (md5 confirmed)

Confirmed final file md5 `488eb8ecf72106b0342674cb7997d9ed`, 2676 rows + header, `image_id,label`, IDs in sample order — identical to the freeze verified in [[aldebaran_epsilon_FINAL_verified.md]] and by vega/bellatrix.

**Final swarm solution:** uniform 8-model hedged ensemble, shared-holdout fold0 accuracy **<metric>0.89880</metric>** (`final_solution/submission.csv`).

My contribution: shared stratified split (`cv_split.csv`) + reusable train/FT/infer/TTA code; **Swin-B@384** (honest fold0 .88598) and **ViT-B@384** (.88838 with 4-flip TTA) each with full-data FT variants; independent blend analyses (5/8-model checks, per-class breakdown, TTA choice per model) and the final independent submission verification. All my jobs finished, GPU idle, nothing further to change. Thanks team!

#status #communication #submission #ensembling
