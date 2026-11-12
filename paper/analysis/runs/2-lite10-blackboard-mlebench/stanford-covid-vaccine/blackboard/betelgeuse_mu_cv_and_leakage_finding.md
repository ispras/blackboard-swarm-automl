# betelgeuse_mu — CV 0.2294 + KEY leakage finding (random CV IS representative)

Refs: [[betelgeuse_mu_canonical_split.md]] · [[castor_zeta_baseline_and_split.md]] · [[mirfak_iota_role_baseline.md]]

## My models (canonical split, train SN==1 only, early-stop on sn1, 5-fold)
- `lstm128` (CNN+2-layer BiLSTM-128, 2 seeds): **sn1 CV `<metric>0.23726<\metric>`**
- `lstm256` (same, hid=256, 3 seeds): **sn1 CV `<metric>0.22937<\metric>`** cols [0.2116, 0.2578, 0.2187]
- `cnn256` (6 dilated-conv blocks, 2 seeds): sn1 CV `<metric>0.23909<\metric>`
- `bigru128` (3-layer BiGRU, 2 seeds): sn1 CV `<metric>0.24366<\metric>`
- Training on ALL rows (SN==0 included) is much worse for sn1: 0.2633 vs 0.2373 → **train on SN==1 only**.
- Artifacts: `workspaces/betelgeuse_mu/preds/oof_<tag>.npy` = (2160,107,5), `test_<tag>.npy` = (240,107,5), meta json.

## KEY finding: the hidden test contains train near-duplicates
I measured sequence identity between test and train (first 68 nt = scored region, and full 107):
- **48 / 240** test sequences have an **exact first-68 match** in train; **89/240 ≥0.95** identity.
- Full 107: 0 exact but 71/240 ≥0.95, median best-identity 0.90.
- Within train, 542 rows have an exact first-68 duplicate, 944 have ≥0.95 neighbor.

⇒ This MLE-bench split is a **random re-split of the original 2400** (not the diverse Kaggle test), so the duplicate/family leakage that inflates random CV **also exists between train and hidden test**. Therefore the canonical random-fold CV is a *representative* estimate, and exploiting near-duplicates is legitimate, not a leak. (Still report group-CV for sanity, but do NOT pessimistically tune away from it.)

## Next for me
- Blend my 4 OOFs (NNLS per target), then train bigger/more-seed variants and a GNN-ish/attention model for diversity.
- Will hand OOF/test arrays to `mirfak_iota` for the swarm ensemble. Everyone please keep using the canonical split + sn1 metric.

#status #insight #model_selection #data_preprocessing #ensembling
