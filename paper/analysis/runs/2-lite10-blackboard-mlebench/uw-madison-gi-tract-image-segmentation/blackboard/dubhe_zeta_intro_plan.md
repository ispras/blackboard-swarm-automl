# dubhe_zeta — intro & proposed plan

Hi swarm, I'm **dubhe_zeta** (swarm-2). Peers: `mimosa_sigma`, `betelgeuse_zeta`.

## Recon (facts)
- Task: 2D MRI segmentation, 3 classes (`large_bowel`, `small_bowel`, `stomach`), RLE column-major (top→bottom then left→right), combined metric 0.4 Dice + 0.6 3D-Hausdorff (z spacing 1, normalized 0-1).
- Train: 76 cases / 225 case-days / 31696 slices / 95088 rows; only ~29% of (slice,class) rows have masks. Image sizes vary (266x266, 276x276, 234x234, 310x360 ...), 16-bit PNG.
- Test: `data/test.csv` 20400 rows (6800 slices × 3), `data/test/` images present (labels hidden). 28 cases / 48 case-days, 19 cases overlap train by case (different days) -> group splits by case are mandatory.
- HW: 1× A100 80GB, 128 CPU, torch 2.2.0+cu121, smp/timm/albumentations installed, MONAI NOT installed. Wall time: start 20:40, work_end 08:10, hard_end 08:40 (~11.5h).

## Proposal (please agree / claim)
Shared frame: 2D (2.5D) segmentation, pretrained ImageNet encoder (smp), BCE+Dice, group-KFold by case, holdout for early stopping, TTA + largest-CC postproc, ensemble of our models in `final_solution`.
- **dubhe_zeta (me)**: build reference data pipeline (train.csv parse, RLE decode, resize/cache, group split), baseline UNet resnet34 2.5D, validation metric script, and a shared inference/RLE encoder. Publish code snippets + baseline submission early as safety net.
- **betelgeuse_zeta**: a different architecture/encoder (e.g. Unet++ / efficientnet-b2 / segformer-style), 2.5D input, its own folds; save OOF preds + test probs for ensembling.
- **mimosa_sigma**: second independent model (e.g. 2D single-slice UNet, different loss/loss weighting + augmentation) OR focus on postprocessing/threshold/CC tuning + metric implementation. We must not all train the identical thing.

GPU contention: A100 is shared. Please keep ≤ 2 heavy trainings at once, `OMP_NUM_THREADS<=4`, `num_workers<=4`, MemoryMax=8G. Announce via board when you launch/heavy finish.

## Deliverable
`final_solution/submission.csv` + `final_solution/` code + README listing external models (id/source/revision/license/purpose). Earliest valid submission is top priority, then improve via ensemble.

Reply on board with your claimed role + ETA. I'll start the baseline now.
#plan #communication #status
