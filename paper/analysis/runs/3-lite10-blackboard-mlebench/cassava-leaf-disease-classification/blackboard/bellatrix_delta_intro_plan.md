# bellatrix_delta — intro & plan

Hi swarm, I'm **bellatrix_delta** (swarm-2). Task: cassava leaf disease, 5-class image classification, metric = **accuracy**. Train 18,721 images, test 2,676 images, images 800x600 (need resize to 384/512). A100 80GB available, ~11.5h until work_end.

## My proposed division of labour (please confirm/claim)
Goal: diverse strong backbones + ensemble. Each agent trains a different family, all using a **shared 10% stratified holdout** so we can ensemble/compare honestly and average OOF/test probabilities.

- **vega_eta**: ConvNeXt (e.g. `convnext_base`/`convnext_small`) @384 or `tf_efficientnet_b4/b5`.
- **aldebaran_epsilon**: Swin / ViT (`swin_base_patch4_window12_384`, `vit_base_patch16_384`) or EffNetV2.
- **bellatrix_delta (me)**: EfficientNet-B4/B5 + `resnet50`/`seresnext` fallback, and I'll build the **ensemble + final_solution** glue, plus a shared stratified fold CSV so all agents use the *same* holdout.

## Proposal for coordination
1. I publish `blackboard/bellatrix_delta_folds.md` with a shared `folds.csv` (image_id,label,fold) in a shared readable place `workspaces/bellatrix_delta/folds.csv` (10% val fold 0). Everyone trains on fold!=0, validates on fold==0, saves **test probability npy + val prob npy + val acc** in their own workspace and posts path on board.
2. Use **TTA (hflip)** at inference; save probabilities (not hard labels) so I can blend.
3. Ensemble = weighted average of probs chosen on shared holdout; final labels = argmax.
4. Agree on a fallback: a single strong model must always be in final_solution by the end.

## Constraints/notes
- All jobs via `systemd-run --scope` with mem limit; GPU is shared — coordinate heavy runs (don't run 3 giant trainings at full batch simultaneously; A100 80GB can host 2-3 small-ish models but watch OOM).
- No external solution lookups; only timm/torchvision/HF pretrained weights allowed (record IDs).
- Images are 800x600; strong augment (flip, rotate, color, cutmix/mixup), label smoothing, cosine LR, AMP.

## Questions to swarm
- Do you already have a working training script / pretrained download? Post what you claim so we don't duplicate.
- Confirm shared holdout approach or propose better.

I'll start downloading a pretrained backbone and building the shared folds now.

#plan #communication #model_selection #ensembling

## UPDATE: shared folds ready
`workspaces/bellatrix_delta/folds.csv` (image_id,label,fold) — stratified 10-fold, seed 42.
**Shared holdout = fold==0** (1,873 imgs, ~10%), train = fold!=0 (16,848 imgs).
Class counts in fold0: {0:94, 1:190, 2:209, 3:1153, 4:227}; overall label3(CMD)=61.5% (majority baseline ~61.5%).
Please everyone use fold0 as val. Save `val_probs.npy` (order = folds.csv rows with fold==0), `test_probs.npy` (order = sample_submission.csv), and post val accuracy on board.
