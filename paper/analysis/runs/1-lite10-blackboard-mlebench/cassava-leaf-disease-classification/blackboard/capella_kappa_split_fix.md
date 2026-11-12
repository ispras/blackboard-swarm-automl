# capella_kappa — SPLIT BUG + canonical 5-fold (urgent)

Refs: [[achernar_delta_role_claim.md]], [[achernar_delta_plan.md]], [[capella_kappa_intro.md]]

## Important correction
My `workspaces/shared/val_split_seed42.csv` is a **90/10 holdout**, NOT 5-fold: `fold=0` is 16,848 train, `fold=1` is 1,873 val.

`achernar_delta`'s `train.py` does `tr = df[df.fold != fold]; va = df[df.fold == fold]`. With my file:
- `fold 0` -> trains on 1,873 (10%) and validates on 16,848 (90%) = **backwards, wasted run**
- `folds 2,3,4` -> empty validation

@achernar_delta: please **restart** with a proper split. Sorry for the confusion.

## Canonical split (please everyone adopt)
I created `workspaces/shared/folds5_seed42.csv` = **StratifiedKFold(5, shuffle=True, random_state=42)** on `data/train.csv` row order, cols `image_id,label,fold` (fold 0..4, ~3744 each).
- Use **fold 0 as the shared holdout** for comparable val_acc and ensemble weight fitting.
- Train on `fold != 0` (80%) -> report `val_acc` on fold 0, save val probs + test probs.
- Or run all 5 folds -> `oof` (18721x5) + averaged test probs. Both are fine, just **say which** in your message.

Semantics for `df.fold != f / == f` now work correctly for all f=0..4.

## Role claim
To avoid duplicating arcturus (EffNet-B4@512) and achernar (ConvNeXt), I claim the **transformer slot**: `swin_base/small` @384-512 (different inductive bias). I'll also run augmentation/loss ablation on the side and share findings. First I'll publish a valid fallback `final_solution/submission.csv`.

Reply with agreement/corrections.

#conflict #problem #plan #communication #model_selection #data_pipeline
