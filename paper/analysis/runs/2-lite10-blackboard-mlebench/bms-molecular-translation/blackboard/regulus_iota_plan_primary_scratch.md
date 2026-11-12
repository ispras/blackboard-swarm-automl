# regulus_iota — scratch model now Lev <metric>17.80<\metric> @ 4k steps (30 min) and inference is cheap: proposal to make it a primary lane

Refs: [[regulus_iota_scratch_2k_lev37.md]] [[regulus_iota_critical_predict_bug_fix.md]] [[altair_theta_throughput_and_scheduling.md]]

## Updated results (same shared 20k holdout, n=300 greedy)
- step 2000: Lev **<metric>37.40<\metric>** (tok acc 0.938)
- step 4000: Lev **<metric>17.80<\metric>** (tok acc 0.978), loss 0.737 @ 5.3k and still falling
- 484,838 test images: full-test inference is **FAST** for this model — measured **44.5 img/s even while training concurrently** (2000 test imgs / 44.9s, KV-cache greedy, bs128) => **~3h full test under contention, ~1.5h if exclusive**.
- MolScribe (corrected transform) is **13-22 Lev** but needs 5-10 exclusive GPU-hours for full test. My model may end up comparable and is 3-6x cheaper to run.

## Proposed GPU schedule (10h left, please object with concrete numbers if you disagree)
- **Now-12:00**: exclusive-ish GPU for my training (short benchmarks OK). At the current trajectory it should reach well below 17.8. This is our highest-ROI use of the GPU right now (metric dropping fast vs hours of inference).
- **12:00-13:30**: I run full-test inference with my best checkpoint -> `workspaces/regulus_iota/preds_scratch.csv` (image_id, InChI, conf, smiles).
- **13:30-15:45**: MolScribe corrected-transform inference on as much of test as fits (full if capped-maxlen benchmark shows <3-4h; else a large shard). @altair_theta owns this; @regulus_omega helps. I will ensemble per-image by confidence (my model + MolScribe), fallback constant where both low-conf.
- **15:45-16:20**: write `final_solution/submission.csv` + solution code + board summary.

## Requests
- @altair_theta: fix `value=(255,255,255)` first (details in [[regulus_iota_critical_predict_bug_fix.md]]), then please benchmark **corrected** transform with capped `FORMAT_INFO['chartok_coords']['max_len']` (try 480/240/192/160) on ~96-512 holdout: report Lev + s/img (exclusive GPU). If 192 is accuracy-safe, full test drops to ~4-5h. Also emit `conf` per image for ensembling.
- @regulus_omega: **please don't start a 4h fine-tune** — its test inference won't fit in the remaining budget. Best use: (a) verify the corrected MolScribe runner end-to-end and produce per-image confidence, (b) chemistry/RDKit postprocessing (graph->SMILES fallbacks, canonicalization, invalid repair), (c) help shard the 13:30-15:45 inference. Your working MolScribe loader is valuable there.

#plan #communication #status #insight #model_selection #inference
