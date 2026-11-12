# Expanded tag vocabulary (v2)

Расширение словаря для **пост-хок LLM-разметки** сообщений. Базовые 22 тега из `PROMPT_RUS.md` сохранены без изменений, чтобы можно было сравнивать старые и новые прогоны. Добавлены 30 specific-тегов с parent-связью и отдельный role-layer на 7 ролей.

**Важно:** это не замена промпта роя. Агентам по-прежнему лучше давать 22 базовых тега (иначе они начнут непоследовательно использовать 50+ тегов и сравнимость сломается). Расширенный словарь нужен для внешней LLM-разметки, где можно контролировать prompt и валидировать результат.

---

## 1. Base tags (22, без изменений)

`data_preprocessing, data_pipeline, feature_engineering, augmentation, architecture, loss, regularization, optimization, training, ensembling, model_selection, postprocessing, inference, plan, communication, submission, status, conflict, problem, bug, insight, other`

---

## 2. Specific tags (30)

### Data and validation

| Tag | Parent | Definition |
|---|---|---|
| `fold_protocol` | `data_pipeline` | Publishing or enforcing a canonical CV split: fold count, seed, row order, class order, and shared fold files. |
| `artifact_io` | `data_pipeline` | Saving, naming, validating, or exchanging OOF/test probability arrays, shapes, row order, caches, and artifact indexes. |
| `leakage_control` | `data_pipeline` | Preventing leakage between base models, meta-models, folds, and submissions; nested or honest validation of stacking. |
| `text_vectorization` | `feature_engineering` | TF-IDF, count vectorizers, word/char n-gram construction, vocabulary limits, and sublinear scaling. |
| `stylometric_features` | `feature_engineering` | Handcrafted text statistics such as length, punctuation, casing, stopword counts, and other stylometric signals. |
| `pseudo_labeling` | `augmentation` | Per-fold teacher/student self-training, confident test soft labels, and semi-supervised expansion. |

### Modeling and training

| Tag | Parent | Definition |
|---|---|---|
| `transformer_finetuning` | `training` | Fine-tuning pretrained language models such as BERT, RoBERTa, DeBERTa, XLNet, ELECTRA, ERNIE, MPNet, BART, or DistilBERT. |
| `classical_linear_model` | `training` | Training TF-IDF/linear/NB baselines such as LogisticRegression, SVM, NB-SVM, ComplementNB, SGD, or Ridge. |
| `tree_model` | `training` | Training gradient-boosted or tree models such as LightGBM, XGBoost, CatBoost, or RandomForest. |
| `backbone_selection` | `architecture` | Choosing a model family, scale, or pretrained checkpoint for diversity or expected gain. |
| `adversarial_training` | `regularization` | Fast Gradient Method, AWP, or related embedding-perturbation training used as regularization. |
| `seed_averaging` | `training` | Running the same configuration with multiple random seeds and averaging predictions. |
| `checkpoint_averaging` | `training` | Averaging predictions across epochs or selecting/averaging best checkpoints within a fold. |
| `hyperparameter_tuning` | `optimization` | Tuning learning rate, batch size, epochs, weight decay, or other optimization hyperparameters. |

### Evaluation and selection

| Tag | Parent | Definition |
|---|---|---|
| `cross_validation` | `model_selection` | Running or reporting k-fold cross-validation and OOF scores. |
| `nested_validation` | `model_selection` | Nested CV, fold-wise weight fitting, or explicit control of stacking/selection optimism. |
| `base_model_diversity` | `model_selection` | Analyzing error decorrelation and prioritizing genuinely different model families, scales, or feature spaces. |

### Ensembling and release

| Tag | Parent | Definition |
|---|---|---|
| `blending` | `ensembling` | Arithmetic, geometric, rank, or greedy weighted averaging of predictions without a trained meta-learner. |
| `stacking` | `ensembling` | Training a meta-learner on concatenated OOF probabilities or log-probabilities. |
| `meta_feature_stacking` | `ensembling` | Stacking with handcrafted meta-features in addition to base-model probabilities. |
| `weight_optimization` | `ensembling` | Fitting blend or stack weights, including greedy search, constrained optimization, and nested weight selection. |
| `calibration` | `postprocessing` | Temperature scaling, probability calibration, or normalization before blending/submission. |
| `submission_validation` | `submission` | Verifying submission ids, row count, class order, probability sums, absence of NaN, and file format. |
| `code_release` | `submission` | Packaging final code, README, reproduction scripts, and artifact indexes. |

### Analysis and coordination

| Tag | Parent | Definition |
|---|---|---|
| `error_analysis` | `insight` | Per-class errors, confusion patterns, residual analysis, or diagnosis of where the model fails. |
| `negative_result` | `problem` | A weak model, failed feature family, or negative ablation that should stop or redirect work. |
| `role_negotiation` | `plan` | Claiming lanes, proposing a division of labor, or negotiating who does what. |
| `gpu_scheduling` | `communication` | Claiming GPU lanes, posting lock/release signals, and managing compute contention. |
| `ownership_handoff` | `communication` | Transferring or protecting ownership of `final_solution`, shared scripts, or submission writes. |
| `conflict_resolution` | `conflict` | Resolving disagreements, write races, duplicated work, or ownership conflicts. |

---

## 3. Role layer (7, отдельно от тегов)

Role-теги не смешиваются с SI/JSD по контенту. Они нужны для прямого ответа на вопрос «какие роли возникли».

| Role | Definition |
|---|---|
| `role_integrator` | Owns blending/stacking, combines peer artifacts, and drives the final submission. |
| `role_base_model_producer` | Produces diverse base models (transformer, classical, tree) that others can stack. |
| `role_training_technique_specialist` | Develops and transfers training techniques such as FGM, pseudo-labeling, seed averaging, or checkpoint averaging. |
| `role_validation_owner` | Owns fold protocol, nested validation, leakage control, and score integrity. |
| `role_coordinator` | Coordinates lanes, compute, ownership, deadlines, and handoffs. |
| `role_analyst` | Produces error analysis, negative results, and insight that redirect the swarm. |
| `role_releaser` | Owns submission validation, final verification, and code release. |

---

## 4. Prompt rules for post-hoc LLM coding

1. Assign **1–4 specific tags** per message.
2. Prefer the **most specific** applicable tag; add the parent base tag only if the message also covers the general category beyond the specific tag.
3. Use `role_*` only when the message's **primary contribution** is that role, not merely because the author has that role.
4. Do not infer tags from the agent identity; use the message content.
5. Use `other` only when no tag fits.
6. For analysis, **roll specific tags up to parents**; keep role tags in a separate layer and do not mix them into SI/JSD.

### Готовый system prompt (EN)

```text
You are a careful annotator of multi-agent machine-learning engineering messages.
Assign 1-4 specific tags from the controlled vocabulary to each message.
Prefer the most specific applicable tag. Add the parent base tag only if the
message also covers the general category beyond the specific tag.
Use role_* tags only when the message's primary contribution is that role.
Do not infer tags from the agent's identity; use only the message content.
Use "other" only when no tag fits.
Return JSON only:
{"tags":["..."],"role_tags":["..."],"confidence":0.0,"evidence":"short verbatim phrase"}.
```

### Готовый блок для `PROMPT_RUS.md` (если всё же хочешь дать агентам)

Не рекомендуется вставлять все 30 specific-тегов в промпт роя. Компромисс — добавить только 8 самых различимых:

```text
21. Add tags to messages (`#super_tag`). A message may have one or more tags.
Base list: data_preprocessing, data_pipeline, feature_engineering, augmentation,
architecture, loss, regularization, optimization, training, ensembling,
model_selection, postprocessing, inference, plan, communication, submission,
status, conflict, problem, bug, insight, other.
Optional precise tags (use only if directly relevant):
fold_protocol, transformer_finetuning, classical_linear_model, adversarial_training,
seed_averaging, blending, stacking, submission_validation.
```

---

## 5. Analysis recipe

1. Tag with the expanded vocabulary (specific tags + optional role tags).
2. Compute SI/JSD on **specific tags** (30) — this is the main role-content analysis.
3. Roll specific tags up to **base tags** (22) for comparability with previous runs.
4. Compute role shares separately: fraction of messages with each `role_*` tag per agent.
5. Validate on 30–50 stratified messages with a human annotator; report Cohen's κ.
6. If κ < 0.7, merge or redefine the confused tags before running the full analysis.

## 6. Files

- `expanded_tag_vocabulary.json` — machine-readable version with parent links and rules.
- `expanded_tag_vocabulary.md` — this document.
