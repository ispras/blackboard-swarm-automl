# arcturus_alpha — code + artifacts (reproducibility)

Part of the swarm ensemble for Spooky Author Identification (multi-class log loss).
All artifacts use the canonical folds `StratifiedKFold(5, shuffle=True, random_state=42)`,
class order **[EAP, HPL, MWS]**, row order = original `data/train.csv` / `data/test.csv`.

## Scripts
- `make_folds.py` — canonical 5-fold indices → `folds.npy`.
- `baseline_linear.py` / `feature_search.py` — TF-IDF + linear baselines; case-sensitivity + char-wb/char findings.
- `make_models.py` — final classical suite (LR/ComplementNB/MNB/SGD/calibrated SVC/PA) on case-sensitive word+char TF-IDF.
- `pseudo_label.py` — leak-free per-fold self-training (`pseudo_lr`).
- `train_tf.py` — HF transformer trainer with **FGM adversarial training** (`--fgm`), epoch-averaging, per-fold checkpoints. Used for `xlnet-base`/`xlnet-large` (+FGM, +seed-avg).
- `make_lgbm.py`, `make_cpu_diverse.py` — negative results (tree/MLP/Nystroem, weak), kept for the record.
- `final_crosscheck.py` — independent read-only stacker sanity check.

## Best base artifacts (OOF logloss) published in `workspaces/arcturus_alpha/` as `oof_X.npy` + `test_X.npy`
| model | OOF |
|---|---|
| `xlnet_large_fgm_seedavg` | 0.29918 |
| `xlnet_large_fgm` / `_s7` | 0.31641 / 0.31471 |
| `xlnet_large_seedavg` | 0.33275 |
| `xlnet_large` / `_epavg` | 0.35235 / 0.34548 |
| `xlnet` / `_epavg` (base) | 0.40018 / 0.39979 |
| `case3_lr` | 0.36707 |
| `pseudo_lr` | 0.36117 |
| `case_cnb` / `case_lr` / `nocase_lr` | 0.369 / 0.370 / 0.375 |

Key contributions: (1) case-sensitivity insight (lowercase=False helps), (2) char_wb+char feature union,
(3) leak-free per-fold pseudo-labeling, (4) FGM applied to the XLNet-large family (0.352→0.316, seed-avg 0.299).
