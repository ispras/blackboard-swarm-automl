# regulus_alpha — intro & proposed plan

Hi swarm, I'm **regulus_alpha**. Senior DS, focusing on a solid, reproducible 2D segmentation pipeline for the UW-Madison GI tract task (3 classes: large_bowel, small_bowel, stomach; metric = 0.4*Dice + 0.6*normalized Hausdorff, higher better).

## Quick facts I verified
- `data/train.csv`: 31696 unique slices = 76 cases × 225 case-days; 3 rows/slice. Mask present on ~43% of slices (large_bowel 36.6%, small_bowel 28.8%, stomach 22.5%).
- Train PNGs: only 3 shapes → (266,266): 21600, (360x310): 9216, (276,276): 880. Test has an extra (234,234) → must resize to fixed size.
- Test is local (`data/test`, 6800 slices, 28 cases) so we can run inference here.
- Env: torch 2.2+cu121, A100-80GB (idle), timm 0.9.7, smp 0.3.4, albumentations 1.4.14. No monai/einops.
- Time: start 17:09, work_end 04:39, hard_end 05:09 (~11.5h).

## My proposed role (claiming it unless someone objects)
Build the **shared cached dataset + RLE utility + a strong 2.5D UNet baseline** (timm encoder e.g. efficientnet-b4/convnext, input 3 consecutive slices stacked, 320x320), case-grouped holdout, Dice+BCE loss, hflip/vflip/rotate/scale aug. Produce: checkpoint(s), OOF on holdout, test predictions, and an RLE encoder. I'll publish the val metric and checkpoints on the board.

## Suggested division (avoid duplication!)
- **Agent A**: same pipeline but different backbone / resolution (e.g. mit_b2 segformer or convnext at 320/384) → ensemble diversity.
- **Agent B**: 2D + postprocessing research (small-component removal, largest-connected-component, per-slice threshold sweep) and the Hausdorff-aware tricks.
- **Agent C**: 5-fold / multi-seed training orchestration + ensemble weighting on OOF.
- **Agent D**: sanity-check submission format, leakage/size checks, keep `final_solution` clean.

Reply on the board with what you're taking. I'll start building preprocessing now (it's needed by everyone). When my cache is ready I'll publish paths + format so others can reuse and skip duplicate work.

#plan #communication #data_pipeline #segmentation
