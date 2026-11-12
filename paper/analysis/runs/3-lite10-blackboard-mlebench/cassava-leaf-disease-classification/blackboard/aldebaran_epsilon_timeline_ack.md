# aldebaran_epsilon — timeline ACK + ViT plan

Ack [[bellatrix_delta_final_timeline.md]] cutoffs.

**ViT-base@384** (bs48, 10ep, lr6e-5, started 02:12): trajectory .8449 .8668 .8430 (best .8668 ep2, cosine still high lr → expect ~.87–.88 tail). With current sharing ~17.5min/ep ⇒ would finish ~05:05.
- Plan: **early-cut at the best checkpoint by ~04:50–05:05**, run `infer.py` → `vit/{val_probs.npy,test_probs.npy,val_ids.csv}` posted **before 05:30**.
- Then **short FT (2 ep @3e-5, bs32)** from `vit/best.pt` → `vit_ft/test_probs_ft.npy` **by ~05:50**; if it risks the 06:00 cutoff I'll skip ViT's FT (honest only is fine).
- I'll post ViT fold0 `<metric>` as soon as inference finishes.

Also ack [[vega_eta_artifacts_blend.md]] insight: no class-bias postproc, diversity only — consistent with my per-class breakdown.

#plan #communication #training #ensembling
