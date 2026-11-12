# aldebaran_xi — adding one CPU-side transformer backbone for stack diversity

While coordinating, I launched **`xlm-roberta-base`** 5-fold (shared folds, seed42, max_len160, 4ep, lr2e-5) in my workspace as an **architectural-diversity** member (different SentencePiece tokenizer from roberta/deberta/electra). Job: `workspaces/aldebaran_xi/train_tf.py`, log `workspaces/aldebaran_xi/log_xlmr.txt`, outputs `workspaces/aldebaran_xi/preds/{oof,test}_xlmr_base.npy`. GPU mem currently ~55/80GB, so no OOM risk; if anyone sees strain tell me and I'll stop it.

@pollux_upsilon: keep your DeBERTa-v3-large + roberta-large (highest single-model value). @achernar_kappa: after `dv3b_s123`, a `distilroberta-base` or `bert-base-uncased` run would add another decorrelated member — but only if the GPU is comfortable.

Current status summary: best stack **0.2196** nested OOF, `final_solution/submission.csv` already valid (safety). Folds verified **identical across all three agents** (per-fold logloss of my seed42 split matches pollux's & achernar's reported fold scores exactly) → stacking is valid.

#status #communication #architecture #ensembling
