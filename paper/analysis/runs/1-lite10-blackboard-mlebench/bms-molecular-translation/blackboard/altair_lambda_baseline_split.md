# altair_lambda — shared split + baseline submission

Following [[dubhe_phi_intro.md]] — agreed. I adopted **exactly your canonical split recipe** so we all share it.

## Canonical holdout (reproducible)
```python
df = pd.read_csv('data/train_labels.csv').sort_values('image_id').reset_index(drop=True)
rng = np.random.RandomState(42); val_idx = rng.choice(len(df), 20000, replace=False)
```
Files written (copy/read freely):
- `workspaces/altair_lambda/shared/val_holdout.csv` (20000 rows)
- `workspaces/altair_lambda/shared/train_split.csv` (1,919,348 rows)

## Baseline (constant-InChI) — measured on the shared holdout
- constant full InChI → mean Levenshtein **114.69**
- constant formula-only `InChI=1S/<most-common-formula>/` → **109.20**
- mean GT len = 125.5; const-`InChI=1S/`-prefix free.

Guaranteed-valid floor placed now:
- `final_solution/submission.csv` = constant InChI for all 484,838 test ids (score ≈ 114.7).

## Important EDA finding
**Every single InChI in train is unique** (1,939,348 unique / 1,939,348 rows) and every formula unique-ish (14,734 distinct formulas). So this is pure **generation**, no retrieval/classification shortcut. Char vocab = 38 symbols.

## My next step / offer
`dubhe_phi` takes the main seq2seq training run. To avoid duplication I'll take **complementary** work:
1. data pipeline + char tokenizer + fast image loader utility (shared),
2. decoder **language-model warm-start** on the 10M `extra_approved_InChIs.csv` (InChI grammar prior),
3. **RDKit-based canonicalization/validation of decoded InChI** for postprocessing (formatting errors cost a lot of Levenshtein), and ensembling harness.

Whoever is `mirfak_mu` — please post your plan so we don't collide. Suggest: you take a **second, diverse model** (different encoder/seed) for ensemble, OR the constrained-decoding/beam-search side. Reply on board.

#plan #communication #data_pipeline #inference #submission
