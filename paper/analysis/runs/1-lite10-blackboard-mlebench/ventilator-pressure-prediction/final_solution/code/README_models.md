# Ensemble member models (final submission)

All models are **trained from scratch on `data/train.csv`** (no pretrained/external weights, no external data, no test labels). Validation: random 10% of breaths, seed 42 (6,790 breaths), scored on inspiratory (`u_out==0`) rows only. Weights fit with multi-start SLSQP (simplex) minimizing holdout inspiratory MAE. For test, each member uses its **full-data retrain** (same config, 100% train, low-LR fine-tune) where available, at the same weight.

This was a **3-agent swarm** (`sirius_tau`, `dubhe_phi`, `antares_nu`). "dubhe"/"antares" members below are peer agents' models trained within this same task on the same `data/train.csv` (not external checkpoints), published under `workspaces/<agent>/preds/`.

| member | arch / features | holdout insp MAE | weight | test preds |
|---|---|---|---|---|
| dubhe_blstm512ft5 | 3×512 BiLSTM, 46-feat cache, repeated low-LR cosine refine | 0.17291 | 0.352 | full-data retrain (`blstm512ft5full`) |
| sirius_c2_lstm3x512_refE | 3×512 BiLSTM, 46-feat cache2, low-LR refine | 0.17988 | 0.178 | full-data fine-tune |
| sirius_c2_lstm3x512_s777 | 3×512 BiLSTM, cache2, seed 777, cosine 145 ep | 0.17853 | 0.167 | full-data fine-tune |
| sirius_c3_lstm3x512_s888 | 3×512 BiLSTM, 59-feat cache3, seed 888 | 0.18165 | 0.093 | full-data fine-tune |
| sirius_c2_lstm384_refS | 4×384 BiLSTM, cache2, low-LR refine | 0.18237 | 0.072 | full-data fine-tune |
| antares_lstm_c2_512_ft | 4×512 BiLSTM, cache2, refine | 0.18937 | 0.065 | full-data retrain |
| sirius_c3_lstm448_s4242 | 4×448 BiLSTM, 59-feat cache3 | 0.18939 | 0.056 | full-data fine-tune |
| sirius_c3_lstm448_refF | 4×448 BiLSTM, cache3, refine | 0.18885 | 0.016 | full-data fine-tune |
| antares_lstm_c2_448_s99 | 4×448 BiLSTM, cache2, seed 99 | 0.19960 | 0.001 | validated |

Members with ~0 weight (published in the pool, dropped): antares `lstm_c2_s21_ft`, `tcn_c2`, `tcn_c2_b`, `gru_c2`, `bilstm320`, dubhe `cnn_lstm`, `blstm512ft3`, sirius `c2_lstm512_cos`, `c2_lstm448_s2024`, `c2_lstm3x512_refE2`.

## Holdout result
- Blended **inspiratory MAE (canonical holdout) = 0.16037** (honest cross-fitted estimate = 0.16089).
- Expiratory rows (`u_out==1`) predicted by a separate blend of exp-capable members: exp MAE = 0.15417 (free hedge; only inspiratory rows are scored per `TASK.md`).

## Sources / licenses
No external pretrained models or datasets were used. Everything derives from `data/train.csv` + `data/test.csv` (features only). Libraries: PyTorch 2.2.0, NumPy, pandas, SciPy, scikit-learn.
