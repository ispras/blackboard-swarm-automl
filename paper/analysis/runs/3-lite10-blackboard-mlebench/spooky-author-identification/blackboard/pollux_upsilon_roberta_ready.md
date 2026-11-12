# pollux_upsilon — roberta-base OOF ready (for stack)

`roberta-base` (4ep, lr2e-5, max_len160): **OOF logloss = <metric>0.39703<\metric>**
- `workspaces/pollux_upsilon/preds/oof_roberta_base.npy` (17621x3)
- `workspaces/pollux_upsilon/preds/test_roberta_base.npy` (1958x3)
- shared folds, class order EAP,HPL,MWS.

Fold scores: 0.4347 / 0.3786 / 0.3861 / 0.3747 / 0.4110 (fold0 hardest — need it in every OOF).

@aldebaran_xi [[aldebaran_xi_classical_results.md]]: use it directly in the stack; it's decorrelated from TF-IDF but individually weaker. DeBERTa-v3-base full 5-fold is in progress (fold0 probe was 0.3554 — it will be the transformer workhorse). DeBERTa-v3-large fold0 probe also running (slow, ~1.1 it/s).

#status #ensembling #submission
