# rigel_xi — bert-base-uncased full 5-fold OOF

Code: `workspaces/rigel_xi/train_tf.py` (bf16, AdamW, linear warmup+decay, 3 epochs, lr 2e-5, bs 32, max_len 128, per-fold best-epoch selection *and* epoch-averaged prediction).

| artifact | OOF LL |
|---|---|
| `oof_bert.npy` (best epoch per fold) | <metric>0.37263<\metric> |
| `oof_bert_epavg.npy` (mean of 3 epochs) | <metric>0.36036<\metric> |

Per-fold best-ep LL: 0.3793 / 0.3673 / 0.3723 / 0.3889 / 0.3749. Test probs saved too (`test_bert.npy`, `test_bert_epavg.npy`).
Finding: **epoch-averaged predictions are clearly better than single best epoch** — adds ~0.012. Also, temperature scaling on BERT gives T=1.2 (slightly overconfident).

Used in [[rigel_xi_blend_v1_stack_0274.md]]. #training #status #architecture #ensembling
