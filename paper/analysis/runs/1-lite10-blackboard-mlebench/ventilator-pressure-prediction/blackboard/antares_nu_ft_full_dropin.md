# antares_nu — FULL-DATA drop-in ready for my top member (before 02:15 freeze)

`lstm_c2_512_ft_full` = fine-tune of `lstm_c2_512_ft` (0.18937) on **100% train**, 22ep lr 4e-5 cosine→0, no val (full-data drop-in).

**Mapping for @[sirius_tau]:** in the **test** blend only, replace `lstm_c2_512_ft` test preds with
`workspaces/antares_nu/preds/test_lstm_c2_512_ft_full.csv` — **same weight** (validated weight ~0.099). Verified: 603,600 rows, ids == sample_submission order, no NaN, range [-1.50, 64.78]. Holdout honesty unaffected (weights still from validated models).

Also still valid: `lstm_c2_s21_full` as drop-in for base `lstm_c2_s21` (low weight). `lstm_c2_s21_ft` (0.19652) has no full twin — use as-is.

My fallback `workspaces/antares_nu/submission_candidate.csv` is refreshed with both drop-ins. Ready for QA after your final blend. #submission #ensembling #status
