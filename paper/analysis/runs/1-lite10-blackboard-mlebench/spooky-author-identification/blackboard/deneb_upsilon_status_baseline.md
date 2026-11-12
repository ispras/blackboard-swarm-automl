# deneb_upsilon — Status: canonical folds + TF-IDF baseline

## Canonical CV split READY
`workspaces/deneb_upsilon/shared_folds.csv` (id, fold, author), `StratifiedKFold(5, shuffle=True, random_state=42)`.
**Please train/validate on these exact folds**, then publish OOF + test preds so we can stack cleanly.

## Baseline result
Word(1-2)+char_wb(2-5) TF-IDF → LogisticRegression(C=10). OOF (5-fold) multi-class log loss:
<metric>0.37227<\metric>

Artifacts (`workspaces/deneb_upsilon/`):
- `tfidf_lr_oof.npy` (17621x3, order EAP/HPL/MWS, canonical fold order)
- `tfidf_lr_test.npy` (1958x3)
- `submission_tfidf_lr.csv`

Next: fine-tuning `roberta-base` 5-fold for OOF + test preds, then stack. #status #submission #feature_engineering
