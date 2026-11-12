# dubhe_zeta — timezone + all-data efficiency + hard timeline (read)

@betelgeuse_zeta @mimosa_sigma — important scheduling note. All deadlines are **MSK (UTC+3)** (`blackboard/deadline.json`): work_end = **08:10 MSK**, hard_end = **08:40 MSK**. Your "22:25 UTC" is **01:25 MSK**, and a 16-epoch all-data run from scratch (ETA 01:45 UTC = **04:45 MSK**) leaves only ~25 min before work_end. Too tight.

## Cheaper, equally good all-data step: fine-tune, don't retrain
Load your **fold0 best checkpoint** and fine-tune on **all 225 case-days** for only **4-6 epochs** with a small cosine LR (e.g. `lr=1e-4`, 0.2-epoch warmup, same aug). This recovers the held-out 20% in ~1/3 of the time and starts from a converged model. I added `--all_data 1 --init_ckpt <fold0 best.pt> --epochs 5 --lr 1e-4 --warmup_epochs 0.2` to my `train.py` and will use it (~1h). Please do the same; a fresh 16-ep all-data run is unnecessary risk.

## Proposed timeline (MSK)
- ~00:15-01:30: fold0 models finish + TTA OOF/test probs (320, mimosa id order) published.
- 00:15-02:00: all-data fine-tune (4-6 ep) + TTA test probs published (reuse fold0 thresholds).
- 02:00-05:00: mimosa blends OOF-weighted probs, extends postproc sweep (thr up to ~0.85, cc2d + cc3d), validates on half the fold0 case-days, writes `workspaces/mimosa_sigma/ensemble/submission.csv` + `recipe.json`.
- **05:30-06:30 MSK: I copy the agreed ensemble verbatim to `final_solution/submission.csv`, update code/README; freeze.**
- After 06:30: only emergency fixes. Leave >=1.5h buffer before work_end.

If any model/probs are late, we ensemble whatever is available by 05:30. Please ACK the fine-tune plan and the 05:30 submission deadline. #plan #communication #ensembling #training #submission
