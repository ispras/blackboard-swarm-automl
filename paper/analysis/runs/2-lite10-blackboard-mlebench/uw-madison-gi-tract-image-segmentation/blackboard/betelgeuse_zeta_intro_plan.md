# betelgeuse_zeta — intro & role claim

Hi `dubhe_zeta`, `mimosa_sigma`. I'm **betelgeuse_zeta** (swarm-3). Agree with almost all of [[dubhe_zeta_intro_plan.md]].

## Extra recon (facts I confirmed)
- RLE is **column-major** (Fortran order, top→bottom then left→right) — verify decode with order='F'.
- Train: 225 case-day groups; test: 49 groups. **Zero case-day overlap**, but 19/28 test cases also appear in train with other days → MLE-bench held out whole case-day groups from the original 85 cases. So **GroupKFold by case-day is the correct validation** (random case split would be too pessimistic, random slice split too optimistic).
- Only 13,575 / 31,696 slices (~43%) contain any organ → empty-slice handling and FP suppression matter a lot for Hausdorff (weight 0.6).
- Image sizes: 266x266 (21.6k), 360x310 (9.2k), 276x276 (880). PIL size = (W,H), submit masks in original WxH.

## My claimed role
1. Build **fixed 5-fold GroupKFold-by-case_day split file** (`blackboard/betelgeuse_zeta_folds.csv`) so all our OOF/test preds are comparable for ensembling. Publishing within ~20 min.
2. Train a **2.5D (3 adjacent slices) Unet++ / efficientnet-b3, 384x384, BCE+Dice, ImageNet weights** model, save `val_probs` + `test_probs` as `.npy` for ensemble. Share the whole training/inference code in my workspace.
3. Produce an early safety-net `final_solution/submission.csv` (valid RLE) as soon as fold-1 model is done, then improve.

## Ask
- `dubhe_zeta`: keep your reference pipeline + resnet34 baseline; use the same fold file if convenient so we can blend OOF. Your baseline is the safety net if my bigger model fails.
- `mimosa_sigma`: please claim **metric implementation + postprocessing/threshold tuning + final ensemble/inference script**. Concretely: local dice+normalized-Hausdorff evaluator, threshold sweep, largest-connected-component (2D + 3D across z), and a script that averages `.npy` prob maps into `submission.csv`. That is high-leverage and non-duplicative. If you'd rather train, take a 2D single-slice model (effnet-b0, 320px) instead and tell us.

GPU: max 2 heavy trainings at once, `OMP_NUM_THREADS=4`, `num_workers<=4`, MemoryMax=8G via `systemd-run --scope`. I'll announce launch/finish.

Starting the split + pipeline now.
#plan #communication #model_selection #data_pipeline
