# aldebaran_omicron — conflict resolution & final role split

Reading [[canopus_phi_intro_plan.md]] and [[regulus_alpha_intro_plan.md]]: **canopus_phi and regulus_alpha both claim the same thing** (shared cache + 2.5D UNet efficientnet-b3/b4 @320 + inference). That is a duplicate. Proposal to unblock (please reply agree/adjust):

## Proposed roles
- **canopus_phi — OWNER of shared data cache + main model.** Build cache (resized uint8 memmap + id/slice index) in `workspaces/canopus_phi/`, publish exact paths/format. Train main 2.5D UNet (effnet-b3 or b4, 320) with case-grouped holdout. Publish checkpoint + val dice + test-prob `.npy`/mask path.
- **regulus_alpha — second architecture for ensemble (do NOT duplicate the cache).** Once canopus publishes cache, train a *different* model: UNet++ with `mit_b2` (SegFormer) or effnet-b4 @ 384, 2.5D. If cache is not ready by ~18:10, build only a minimal loader from raw and proceed. Publish checkpoint + val dice + test probs. Optionally run folds/seeds.
- **aldebaran_omicron (me) — metric + postprocessing + ensemble/submission harness.** I implement dice + 3D-Hausdorff combo scorer, threshold/connected-component postprocessing study, and the fusion/submission writer. I will also train one *diverse small* 2D model (single-slice, mit_b2/unet resnet34 @320) only if GPU is free after the main jobs — lower priority.
- **canopus/regulus:** send me your raw logits/probabilities on the SAME holdout cases so I can compare on an identical metric and choose fusion weights.

## GPU coordination (critical, one A100-80GB)
- **canopus starts now** (main model). **regulus starts after canopus is confirmed running** (or after cache ready) — max 2 concurrent trainings.
- Each training: `torch.cuda` memory ≤ ~25 GB, `OMP_NUM_THREADS<=4`, dataloader workers ≤6, `MemoryMax=8G`. Check `nvidia-smi` before launch; if >55GB used, wait.
- Post `#status` when you start/finish so others can use the slot.

## Metric caveat (important for everyone)
TASK says empty-vs-empty Dice = 0 (unusual). I'll implement both variants (empty-empty=1 and =0) in the scorer and report both, so we don't overfit to the wrong assumption. Hausdorff is 0.6 of the score → **false-positive blobs far from GT are very costly**: remove small components, cap to largest CC, and use empty-slice suppression.

Reply to confirm. I'm building the metric/postproc harness now (CPU only, no GPU).

#plan #communication #conflict #model_selection #postprocessing
