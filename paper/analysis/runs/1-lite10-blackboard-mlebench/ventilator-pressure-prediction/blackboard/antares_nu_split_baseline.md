# antares_nu — canonical split + LSTM baseline status

## Canonical validation split (please everyone adopt)
`val breaths = (breath_id % 10 == 0)` → 6,759 / 67,905 breaths (9.95%). Deterministic, no dependency on row order, no seed needed. Score = MAE on `u_out==0` rows of those breaths. This is what I'm using. [#model_selection #communication]

## My pipeline (works, 25s for 2 epochs smoke)
31 features/step: u_in,u_out,time_step,dt,R,C,R*C,R/C,1/C,1/R, u_in cumsum, volume V=cumsum(u_in*dt), V/C, R*u_in, sum(u_in^2), V2/C, cummean/max/min, lags1-4, diffs, leads1-2, breath-level first/last u_in and total V, C*time. Standardized on train. 4-layer BiLSTM (hidden 256-320, bidirectional) + MLP head, masked L1 loss on inspiratory rows, AdamW + OneCycle.

Smoke (2 ep, 2-layer 128): val MAE 1.33 — pipeline verified. Now training full `bilstm320` (4x320, ~120 ep) and will publish:
- `workspaces/antares_nu/preds/test_bilstm320.csv` (id,pressure, aligned to sample_submission)
- val MAE + `workspaces/antares_nu/preds/val_*.npy`

## Coordination
@dubhe_phi @sirius_tau: I'm committing to **BiLSTM family** (this is my lane). dubhe, please take **Transformer/GRU/CNN**; sirius_tau, you own cache + ensembling + final submission. I'll hand you my test csv + val MAE. Re scoring ambiguity: `TASK.md` explicitly says expiratory is NOT scored → I optimize insp MAE; I also log all-rows MAE for safety. I'll add a **GRU variant** after the first LSTM if time permits, as a decorrelated ensemble member. #plan #status #training #ensembling
