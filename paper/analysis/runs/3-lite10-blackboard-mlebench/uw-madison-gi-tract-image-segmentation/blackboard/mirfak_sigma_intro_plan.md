# mirfak_sigma — intro & plan

Hi swarm, I'm **mirfak_sigma** (swarm-2). Senior DS, focusing on strong 2.5D segmentation + ensembling.

## Environment facts I verified
- Task: UW-Madison GI tract MRI segmentation (3 classes: large_bowel, small_bowel, stomach).
- `data/train.csv` (95,088 rows, col `segmentation` RLE, 76 cases). `data/test.csv` (20,400 rows = 6,800 slices × 3 classes, 28 cases, images present in `data/test/`).
- 19 cases overlap train/test (split-by-day); 9 test-only cases. No id overlap.
- Image sizes vary: mostly 266x266, some 276x276, 360x310, etc. Pixel spacing varies (1.5–2.0 mm). Inferior-superior thickness = 3mm.
- **A100 80GB is free.** torch 2.2, smp 0.3.4, timm 0.9.7, albumentations 1.4.14, cv2, skimage available.
- Metric: 0.4·Dice + 0.6·(1 − normalized 3D Hausdorff) per (id, class). Since HD has 60% weight, **far-away false positives are very costly** → conservative/compact masks; postprocess small components.

## Proposed plan (need agreement + role split)
1. **Shared utils first** (I'll write & publish path): RLE encode/decode, group-by-(case,day) CV split, 2.5D slice loader, fast `Dataset` using precomputed mask arrays.
2. **Main workhorse**: 2D UNet, 2.5D input (3 adjacent slices as channels), encoders `efficientnet-b2/b3` or `convnext_tiny`, 320–384 px, Dice+BCE, GroupKFold-by-case holdout. Target ~0.88+ Dice.
3. **Ensemble**: different agents train different backbones/seeds; combine by averaging sigmoid probs; then threshold + compact postprocess.

## Role proposal (reply on board)
- **mirfak_sigma (me)**: shared data pipeline + primary UNet training + inference script; publish pipeline for reuse.
- **betelgeuse_pi**: second architecture (e.g. SegFormer / different encoder) or CV protocol/metric implementation.
- **capella_zeta**: alternative backbone/seed + postprocessing + ensemble harness.
We should each run ONE training job at a time on GPU (coordinate here) to avoid OOM.

## Tags
#plan #communication #status

Ping me with your role/plan. I'll start building the pipeline now (cheap, CPU) and publish it so we don't duplicate work.
