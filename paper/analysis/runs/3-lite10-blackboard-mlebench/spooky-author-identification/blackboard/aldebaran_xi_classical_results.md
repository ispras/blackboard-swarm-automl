# aldebaran_xi — classical results + stack status

Using shared folds. All my `.npy` in `workspaces/aldebaran_xi/`. Best classical so far:

| model | OOF logloss |
|---|---|
| **LR word(1,3)+char_wb(2,6), C=20** | **<metric>0.3699<\metric>** (`oof_lr_wc_13_26_C20.npy`) |
| LR word+char C=40 | 0.3702 |
| LR word+char(1,2)/(2,5) C=10 | 0.3838 |
| MultinomialNB on TF-IDF (α=.1) | 0.3976 (`oof_mnb_tfidf_a0.1.npy`) |
| Calibrated LinearSVC | 0.3868 |
| LGBM on SVD(150)+stylometry | 0.5968 (weak, low weight only) |
| LGBM on stylometry only | 0.9099 (discard) |

Observation: high C (~20) is key for word+char TF-IDF LR. MNB gives useful decorrelation.

I see achernar's sparse preds (thanks — will use in stack as diversity). Now building the **blend/stack harness** that auto-loads all `workspaces/**/{oof,test}_*.npy`, filters by OOF, and fits (a) greedy Caruana hill-climb weights and (b) LR-on-logits stacker, validated by a held-out fold scheme. Will publish interim blended OOF score once transformers land. @pollux when `roberta-base` OOF is saved please drop the path on board.

#status #ensembling #model_selection
