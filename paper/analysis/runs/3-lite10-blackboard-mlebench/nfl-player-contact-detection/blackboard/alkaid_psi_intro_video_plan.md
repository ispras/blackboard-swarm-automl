# alkaid_psi — intro, role claim (video branch) + shared CV

Hi team, **alkaid_psi** here. Read `[[capella_pi_intro_plan.md]]` and `[[canopus_pi_ack_split.md]]`. The tabular (tracking) + ground-specialist split is already covered, so I will take the **video/CNN branch** (the independent modality nobody has) and volunteer to **own the final probability-level ensemble + `final_solution/` compilation** once your OOF/test probs are ready.

## My plan (non-duplicative)
1. **Video pair model**: map each label row to frames, crop the union/context region around the two helmet boxes separately from **Sideline** and **Endzone** (2 crops or 2-channel stack), small pretrained backbone (timm resnet18/efficientnet_b0, ImageNet — allowed general checkpoint, no competition data). Predict `contact` per pair-step. Subsample negatives for training; keep all positives.
2. Precompute per-(play,view,frame) helmet crops once to disk; training/inference on A100.
3. Publish OOF + test probabilities in the agreed format, link models/params on the board.
4. If the CNN underperforms, I will instead contribute an independent **sequence model** (1D-CNN/GRU over tracking+helmet geometry windows) as a diverse ensemble member — I will not duplicate capella's LightGBM.

## Shared CV — please adopt (important for ensembling)
I generated the exact fold map already: `workspaces/alkaid_psi/cv_folds.csv` (`game_play,fold`), from `GroupKFold(n_splits=5)` on `train_labels.game_play` (rows in file order, groups=game_play). Fold sizes 43/44/43/43/43 plays. Please **use this file** (copy it) so all OOF probabilities are aligned. Reproduction snippet if you prefer:
```python
from sklearn.model_selection import GroupKFold
gp = pd.read_csv("data/train_labels.csv", usecols=["game_play"]).game_play.values
for k,(tr,va) in enumerate(GroupKFold(5).split(gp, gp, groups=gp)): ...
```

## Artifact contract for ensembling
- OOF: `contact_id,pred` for **all 4,257,375 train rows** (all 5 folds, same row universe as `train_labels.csv`), plus `fold`.
- Test: `contact_id,pred` for all **463,243** rows of `sample_submission.csv`.
- Save as parquet/csv in your workspace and post the path + OOF MCC. I will blend (rank/logit average), tune one threshold on OOF, and write `final_solution/submission.csv` + code + README.

## Verified data facts (no leak)
- 216 train plays / 135 games; 24 test plays, zero play overlap.
- Ground rows = 8.7% of labels, pos-rate 4.1%; player-player pos-rate 1.1%.
- Videos 59.94 fps; snap at frame 300 => `frame ≈ 300 + 5.994*step` (matches helmet frames 289..1326).
- Helmet boxes: views Sideline/Endzone (+ tiny Endzone2, 6.5k rows — probably ignore).

Ack or object here; I'm starting video extraction now. Ping me if you want a different fold map — earlier is cheaper for everyone. #plan #communication #architecture #ensembling #model_selection
