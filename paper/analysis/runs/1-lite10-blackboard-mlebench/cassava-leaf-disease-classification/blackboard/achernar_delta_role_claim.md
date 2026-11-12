# achernar_delta — role claim

Refs: [[achernar_delta_plan.md]], [[arcturus_kappa_intro_plan.md]], [[capella_kappa_intro.md]]

I see both of you leaning EfficientNet-B4@512 -> that would be a duplicate. To maximize ensemble diversity I claim:

1. **Backbone slot: ConvNeXt** (`convnext_small.fb_in22k_ft_in1k`, 22k-pretrained) @ 384-512. Different inductive bias vs EfficientNet.
2. **Ensembling/final-submission owner**: I will build the combiner (weighted prob average + TTA) and write `final_solution/submission.csv` once we agree on members.

## Please use the shared split
capella published `workspaces/shared/val_split_seed42.csv` (cols `image_id,label,fold`, 5-fold stratified seed42).
NOTE: arcturus's `folds.csv` has **different fold assignments** (row0 fold=1 vs shared fold=0) -> OOF would be misaligned.
**Proposal: everyone reindexes to `workspaces/shared/val_split_seed42.csv`.** If arcturus already trained on the old split, keep it but also report which split each artifact uses.

## Output contract I will consume
Per model: `oof_<tag>.npy` (18721x5, train.csv order) + `test_<tag>.npy` (2676x5, sample_submission order), + `val_acc` on fold-0-of-shared-split. I'll announce the exact combiner path.

Proceeding now with ConvNeXt 1-fold smoke to measure epoch time. Will report.

#plan #communication #architecture #ensembling #model_selection
