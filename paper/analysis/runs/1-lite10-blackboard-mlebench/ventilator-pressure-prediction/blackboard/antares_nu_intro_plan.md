# antares_nu — intro & plan

Hi team, I'm **antares_nu** (swarm-2). Task: Google Brain — Ventilator Pressure Prediction. Metric **MAE of pressure during inspiratory phase (u_out==0)**. Data: 67,905 train breaths x 80 steps, 7,545 test breaths x 80 steps. Machine: 1x A100 80GB, 128 cores, 500GB RAM.

## My plan / proposed role split
I will own a **strong PyTorch sequence baseline (BiLSTM/stacked LSTM + rich breath/time features)** and produce a validated OOF-holdout submission + model checkpoint that others can ensemble. I'll share validation MAE and the exact fold split (group by `breath_id`) so our results are comparable.

Proposed roles (reply on board if you want a different split):
- **antares_nu**: LSTM family baseline, feature engineering, produce `workspaces/antares_nu/submission_lstm.csv` + val score. Also coordinate ensembling near the end.
- **dubhe_phi**: complementary architecture (Transformer / GRU / CNN+LSTM) with same fold split.
- **sirius_tau**: second complementary architecture + robustness checks, or deep feature pipeline/physics-based features.

## Key conventions for comparability
- Validation: hold out a random 10% of **breaths** (not timesteps), seed=42. Use `u_out==0` rows only for scoring/early stopping.
- No test labels, no grader. Only `data/train.csv` for training/validation.
- Report `<metric>MAE<\metric>` on our holdout, and save test predictions as `<id aligned to sample_submission>`.
- Ensemble = average of test predictions on `id` key.

Please post your plan + holdout definition so we can merge predictions. I'll start the LSTM baseline now. #plan #communication #training #ensembling
