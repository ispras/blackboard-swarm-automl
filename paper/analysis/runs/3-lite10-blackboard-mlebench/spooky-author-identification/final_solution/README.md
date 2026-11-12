# Spooky Author Identification — swarm solution (owner: aldebaran_xi)

3-class author classification (EAP / HPL / MWS) on sentence text.
Metric: **multi-class log loss** (lower is better).
Final artifact: `submission.csv` (columns `id,EAP,HPL,MWS`, 1958 rows).

## Final score
**Nested-CV OOF log loss = 0.1861** (mean over 3 independent 5-fold meta-splits, seeds 100/101/102; sd 0.0002).
A 5-seed estimate (incl. seed 42) is 0.1870. See `stack_meta.json` (`nested_oof_logloss`).

## Approach
Two-level ensemble.

### Level 1 — base models (all trained with the same validation partition)
- **Classical sparse text**: TF-IDF word(1-3) + char_wb(2-6) n-grams → LogisticRegression (C sweep), LinearSVC (calibrated), SGD, MultinomialNB, ComplementNB, NB-SVM; LightGBM on SVD(150)+stylometric features.
- **Transformer fine-tunes** (5-fold, fold-averaged test predictions):
  `roberta-base`, `roberta-large` (2 seeds), `microsoft/deberta-v3-base` (3 seeds),
  `microsoft/deberta-v3-large` (3 seeds in final), `google/electra-base-discriminator`,
  `google/electra-large-discriminator`, `bert-large-uncased`, `xlm-roberta-base`,
  `xlm-roberta-large`, `sentence-transformers/all-mpnet-base-v2` (frozen embeddings + linear head).

### Level 2 — stacker
- Input features: `concat(log p, p)` over every base model with solo OOF log loss < 0.65
  (**56 members** in the final model). Two models (`deberta-v3-large` seed555 and
  `bert-large-cased`) were excluded after multi-seed nested-CV showed they were neutral/
  harmful (agreed with an independent teammate audit).
- Model: multinomial `LogisticRegression(C=0.01)`.
- Final test predictions = mean of **20 stackers**, each fit on a random 80% of the OOF matrix
  (variance reduction).

## Validation protocol (train-only, no leakage)
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over `data/train.csv` in original
  row order; class order **EAP,HPL,MWS**. Every agent used exactly this split (verified by
  reproducing each transformer's per-fold log loss).
- Early stopping / epoch selection for every transformer used only its fold's train/val split.
- The stacker hyper-parameters and model inclusion were chosen by **nested** CV on the OOF matrix.
- `train.csv` (17621) + `test.csv` (1958) = 19579 = the original public train, so the hidden test is
  a random ~10% slice and OOF/nested-CV is expected to track it closely.
- No test labels were read at any point; no leaderboard/grading feedback was used.

## Post-processing
None. Temperature scaling (T=1 optimal), prior mixing and probability flooring were all tested on
nested OOF and **hurt**; the raw stacker probabilities are used, as independently reconfirmed.

## Files
- `submission.csv` — final predictions.
- `solution.py` — reproduces `submission.csv` from the base-model prediction matrices
  (`workspaces/*/preds/{oof,test}_*.npy`): discovery, partial-file guard, filter, nested-CV
  estimate, bagged stacker, writes submission.
- `train_transformer.py` — transformer fine-tuning code (5-fold, saves `oof_*.npy`/`test_*.npy`).
- `classical_models.py` — TF-IDF + linear / NB baseline code.
- `stack_meta.json` — nested score, hyper-parameters, exact member list with solo OOF.

## External checkpoints (source: Hugging Face Hub via `transformers`; default `main` revision, 2026-10-30)
| model ID | purpose | license |
|---|---|---|
| `FacebookAI/roberta-base` | base fine-tune (diversity) | MIT |
| `FacebookAI/roberta-large` | base fine-tune (strong member, 2 seeds) | MIT |
| `microsoft/deberta-v3-base` | base fine-tune (3 seeds) | MIT |
| `microsoft/deberta-v3-large` | base fine-tune (strongest family, 3 seeds in final) | MIT |
| `google/electra-base-discriminator` | base fine-tune (diversity) | Apache-2.0 |
| `google/electra-large-discriminator` | base fine-tune (diversity, strong stack member) | Apache-2.0 |
| `google-bert/bert-large-uncased` | base fine-tune (large, decorrelated) | Apache-2.0 |
| `google-bert/bert-large-cased` | base fine-tune (cased tokenizer; tested, excluded from final) | Apache-2.0 |
| `FacebookAI/xlm-roberta-base` | base fine-tune (different tokenizer) | MIT |
| `FacebookAI/xlm-roberta-large` | base fine-tune (large, different tokenizer) | MIT |
| `sentence-transformers/all-mpnet-base-v2` | frozen sentence embeddings → linear head | Apache-2.0 |

All are general-purpose pretrained language models. None was trained on this competition's labels
and none is a competition solution. No external APIs, web search, or competition-specific
checkpoints were used.

## Negative results (recorded)
- Domain-adaptive MLM continued-pretraining on unlabeled train+test text (`dapt_dv3b`) did not help
  the stack (neutral/slightly harmful) and was excluded.
- LGBM / MLP meta-stackers were worse than multinomial LR (0.238/0.234 vs 0.220 at an earlier stage).
- Stylometric/length meta-features did not help.
- Additional DeBERTa-large seeds saturated (~neutral); architecture diversity was what still helped.
