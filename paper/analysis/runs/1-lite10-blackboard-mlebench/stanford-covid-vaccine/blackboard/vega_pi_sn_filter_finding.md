# vega_pi — Key finding: train on SN_filter==1 only (+ baseline)

Agreeing with [[mirfak_zeta_cv_baseline.md]] and [[fomalhaut_beta_intro.md]]:
- Metric = MCRMSE on 3 scored cols. Adopt shared split `[[mirfak_zeta_cv_split.npy]]` (KFold5 seed42, file order). ✅ I am using it.
- Yes: validate/report on `SN_filter==1`.

## 🔑 Finding: filter TRAINING rows to `SN_filter==1` (1349 rows)
`test.json` was quality-filtered (min>-0.5, S/N>1). The `SN_filter==0` train rows have huge outliers (targets to ±44) and inject a lot of label noise. Training **only on the 1349 clean rows** is a large win.

**Same shared split (fold0), my CNN+BiLSTM (`workspaces/vega_pi/cv_train.py`):**
| training rows | fold0 MCRMSE (on SN=1 val) |
|---|---|
| all 2160 | ~0.58 (unfiltered-holdout run) |
| **SN=1 only (1349)** | **<metric>0.2303<\metric>** |

vs mirfak fold0 `<metric>0.3802<\metric>` trained on all rows. The gap is dominated by the noisy-label rows, not architecture.

Reference datasets: mean-predictor on clean rows = 0.49; on all rows = 0.79.

## My plan / role
- **vega_pi**: CNN+BiLSTM (`cnn_lstm`) strong single model + multi-seed/arch **ensemble**; produce 5-fold **OOF (2160,107,5)** + test `(240,107,5)` npy aligned to shared split, on the SN=1 subset.
- Suggest: mirfak = CNN+BiGRU; fomalhaut = BiLSTM/GRU; then we stack/average OOFs.

**Ask:** everyone please re-run your models trained on SN=1-only and compare on the shared split. Then we ensemble OOFs.

#insight #model_selection #data_pipeline #ensembling #training
