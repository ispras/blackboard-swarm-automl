# canopus_iota — contribution to the final ventilator-pressure solution

Agent: `canopus_iota` (swarm member). All my work is reproducible from the scripts below.

## What I contributed
1. **Shared data pipeline** (`canopus_build_shared.py`): built `workspaces/shared/folds.csv` (5-fold GroupKFold seed 42, **fold 0 = common holdout**) and `workspaces/shared/ventilator_v1.npz` (breath tensors `(N,80,29)`: u_in, cumsum/cummean, lags/leads 1-4, diffs, time, dt, u_out, R/C + one-hots, interactions, cumulative volume). This was used by all three agents to keep OOF predictions stackable. Also `workspaces/shared/test_id_sorted.npy` (canonical test row order).
2. **BiLSTM models**: 3x384 5-fold (`lstm384_5f2`, fold-0 insp MAE 0.1965, full 5-fold OOF 0.2005), 3x512 on castor cache2 (`lstm512_c2b`, 0.1997), 58-feature variant (`lstm384_v2f`, 0.2057), an all-data twin of 5f2, and an independent-seed replication of the best RC-EMA recipe (`c2s123` = 0.1796, `e3_s456` = 0.1741, plus all-data twin). These are blend members in the final submission.
3. **Metric semantics + validation protocol**: established that only inspiratory rows are scored (TASK.md; verified against Kaggle-style public results), while keeping a cheap all-rows hedge via loss weight `exp_w≈0.3-0.4`, so predictions are safe under either grader.
4. **Independent blend cross-check** (`canopus_final_blend.py`): phase-split (separate insp/exp) non-negative L1 weights + all-data test substitution; fold-0 cross-fitted CV 0.16062 vs castor's 0.16033/0.1609. Also tested a LightGBM stacker — clearly worse (0.1773), confirming linear/greedy weights are right.
5. **Error analysis**: per-breath MAE is heavy-tailed (median 0.095, worst 1%/5% of breaths = 14%/32% of total abs error, concentrated in R=50 / low-u_in lungs). Documented on the board; motivated castor's RC-EMA physics features (which improved the best single to 0.1716).

## Final submission
`submission.csv` is the swarm's fold-0 cross-fitted blend of fold-0 models with all-data (folds 0-4) test-side twins substituted. Built by `castor_eta`/`bellatrix_iota` with my models included; verified 603,600 rows, ids identical to `sample_submission.csv`, no NaN, clipped to [-2, 65].

## Board messages (chronological highlights)
`canopus_iota_intro_and_plan.md`, `canopus_iota_shared_cache_ready.md`, `canopus_iota_metric_and_status.md`,
`canopus_iota_first_lstm_result.md`, `canopus_iota_group_finding.md`, `canopus_iota_5f2_done_heavytail.md`,
`canopus_iota_blend_0161.md`, `canopus_iota_e3seed_01606.md`.

## Reproduce
```
pyenv shell sis-black
python canopus_build_shared.py            # shared tensors + folds
python canopus_train_lstm.py --folds 0,1,2,3,4 --epochs 200 --hidden 384 --layers 3 --exp_w 0.35 --tag lstm384_5f2
python canopus_final_blend.py             # fold-0 blend + candidate_submission.csv
```
No test labels or external feedback were used at any point; validation/early stopping used only fold 0 of the provided train split.
