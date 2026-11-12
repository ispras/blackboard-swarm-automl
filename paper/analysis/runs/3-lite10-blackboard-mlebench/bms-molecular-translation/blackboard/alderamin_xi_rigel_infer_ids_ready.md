# `rigel_infer_ids.csv` READY (283,168) + safety net now ~32.8; training target 16:40
#inference #plan #status #ensembling

- Created **`workspaces/alderamin_xi/rigel_infer_ids.csv`** = `1m-invalid OR 1m-score<0.7` = **283,168 ids** (covers thresholds up to 0.7; I'll choose 0.4–0.6 on the final holdout). **rigel_mu: use this file** instead of the 97k one so we have rigel preds for the low-confidence-valid rows too. If 283k is too slow, do `invalid OR score<0.5` = 250,246 (tell me and I'll regenerate/label it).
- `final_solution/submission.csv` now = **TTA safety net** (gate + best-score valid rotation on both-invalid): RDKit-valid 95.9%, projected **~32.8**. This will be replaced by the rigel rule (~19–20 projected) after 16:40.
- TTA finished all 3 rotations on the 50,694 both-invalid ids (`preds_tta/`, ~30k rescued). **altair: TTA is done, GPU is free for rigel training — thank you; no more GPU work needed from you.** 
- Current rigel trajectory: 25.9 @step12k, still annealing. 

At ~16:40–17:15 when `rigel_val20k.csv` + `rigel_test` preds land, I will: (1) tune threshold on the 2,000-row paired holdout, (2) build final submission, (3) validate + write README/code. rigel: post your inference throughput/ETA as soon as you start so we can cut the set if needed. Tags: #inference #plan #status
