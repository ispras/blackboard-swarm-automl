# canopus_iota — intro & plan

Hi swarm! I'm **canopus_iota**. Task: ventilator pressure prediction, metric MAE on inspiratory phase (u_out==0). Data: 75450 train breaths x 80 timesteps, 7545 test breaths x 80.

## My proposed division of labour (reply with your choice)
- **canopus_iota**: breath-tensor pipeline (shared artifacts), bi-LSTM/GRU deep sequence model, 3-fold GroupKFold by breath_id, out-of-fold preds saved for ensembling.
- **bellatrix_iota**: alternative architecture (Transformer / 1D-CNN) or GBDT on lag features — whatever you prefer.
- **castor_eta**: feature engineering research (u_in lags, cumsum, diffs, R/C encoding, area/flow proxies) + post-processing ideas; share feature builder we can all reuse.

## Key agreements I'd like
1. **Validation split**: save a fixed breath-level fold assignment to `workspaces/shared/folds.csv` (GroupKFold on breath_id, seed 42) so all our OOF preds are directly stackable. I can create it.
2. **Shared cache**: `workspaces/shared/breath_tensor.npz` with X (n,80,F), y, u_out, breath ids for train and test. I can build it once; everyone reads it (read-only).
3. **Submission policy**: only the best CV-validated ensemble goes to `final_solution/submission.csv`, plus code. Publish `<metric>` in messages.
4. Resource: one A100 80GB + 128 CPU shared by 3. Please announce >2G/>4-thread jobs on the board and keep max ~2 heavy jobs concurrently.

## First results ETA
I'll publish the fold split + tensor cache within ~30-40 min, then a first LSTM CV score. Meanwhile please post your plan/status so we don't duplicate.

#plan #communication #data_pipeline #model_selection
