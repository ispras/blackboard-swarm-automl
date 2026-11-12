# Spooky Author Identification — swarm solution

Task: 3-class (EAP/HPL/MWS) sentence author classification. Metric: **multi-class log loss** (lower better).

Final: `submission.csv` (1958 rows, cols `id,EAP,HPL,MWS`). **OOF ≈ 0.1846** (5-fold, canonical StratifiedKFold seed=42), honest nested meta-combine ≈ 0.1846.

## Pipeline
1. **Base models** trained/validated on the canonical 5-fold split, saving out-of-fold (OOF, 17621×3) and test (1958×3) probabilities, class order [EAP,HPL,MWS].
   - **Transformers** (bf16, AdamW, linear warmup+decay): BERT (base / max_len192 / cased / large-cased), DistilBERT, RoBERTa (base/large), DeBERTa-v3 (base/large/small), ELECTRA, XLNet (base/large), ALBERT, ERNIE, MPNet, BART. Many with **FGM adversarial training** (`--fgm 1.0`) and multi-seed averaging; epoch-averaging (late epochs for FGM).
   - **Classical**: TF-IDF word/char (case-sensitive variants) + LogisticRegression / SGD / ComplementNB / NB-SVM / calibrated SVC / PassiveAggressive; pseudo-labeling LR.
2. **Stacking** (`code_rigel_xi/blend2.py`): per-model temperature calibration → meta-features = concatenated probabilities **and** log-probabilities of all base OOF + handcrafted stylometric features → stackers (LogisticRegression, MLP, LightGBM, seed-averaged) trained with **nested 5-fold CV** on the same folds → pairwise stacker combination (weight fitted on OOF, verified with a nested check).
3. **Meta-combine** (`code_rigel_xi/final_combine.py`): robust weighted average of the unified blend with a second agent's independently-built stacker (fixed/nested-verified), then renormalize rows.

## Files
- `submission.csv` — final probabilities.
- `code_rigel_xi/train_tf.py` — transformer fine-tuning (FGM, epoch-averaging, bf16).
- `code_rigel_xi/blend2.py` — unified stacking/blending + nested validation.
- `code_rigel_xi/final_combine.py` — final meta-combine with nested honesty check.
- `code_rigel_xi/common.py`, `make_seedavg.py` — shared data/CV utilities and seed averaging.

No external data and no test labels were used; validation used only hold-out folds cut from `data/train.csv`.
