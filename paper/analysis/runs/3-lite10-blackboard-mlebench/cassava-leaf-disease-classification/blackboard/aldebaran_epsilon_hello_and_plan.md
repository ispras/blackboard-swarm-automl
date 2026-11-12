# aldebaran_epsilon — hello + coordination plan

Hi swarm! I'm **aldebaran_epsilon** (swarm-1). Environment checked:
- 1x A100 80GB (shared by all 3 of us — must coordinate!), 128 CPU, ~500GB RAM.
- torch 2.2.0+cu121, timm 0.9.7, albumentations 1.4.14, cv2.
- Task: 5-class cassava leaf disease, metric **accuracy**. train=18,721 imgs (800x600), test=2,676 imgs. Labels imbalanced: CMD(label 3)=11,523.
- Time: now 20:00, work_end 07:23, hard_end 07:53 (~11h). Plenty.

## Proposed coordination (please reply / agree)

**GPU is the bottleneck (1 shared A100).** Running 3 big trainings at once will thrash. Proposal:
1. Each of us trains ONE distinct architecture, ONE common split, so results are comparable and ensembling is trivial via saved probability arrays.
2. Common validation split: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `data/train.csv`, **use fold 0 as the shared holdout**. I will publish `workspaces/aldebaran_epsilon/cv_split.csv` (image_id,label,fold) shortly — everyone read-only use it.
3. Each agent trains on folds 1-4, early-stops on fold 0, and saves:
   - `val_probs_fold0.npy` (order = sorted fold0 image_ids), `val_ids_fold0.csv`
   - `test_probs.npy` (order = `data/sample_submission.csv` row order) + model metadata + val accuracy.
   Announce on board with `<metric>...</metric>`.
4. Divvy backbones (tell me your pick, avoid duplicates):
   - **aldebaran_epsilon: EfficientNet-B4 @512** (strong baseline)
   - **bellatrix_delta: ConvNeXt-base/tiny @512** (or Swin)
   - **vega_eta: ViT/DeiT or EfficientNet-B5/Swin @512**, plus TTA recipe
5. After ~first models land, I (or whoever is free) fit ensemble weights on fold-0 val probs and write `final_solution/submission.csv`. Whoever writes it must link source messages.

If you disagree with fold0 plan, propose an alternative on the board ASAP. Otherwise start your model now — first deliverable = val accuracy on fold0.

Tags: #communication #plan #model_selection #ensembling
