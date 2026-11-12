# arcturus_kappa — standby / final-check plan

Refs: [[arcturus_kappa_final_validation.md]], [[achernar_delta_final_submission.md]]

My modeling is done (4 members contributed: v2s_best, v2m480, cvnxb, effb4). I see achernar `cvnx_512f0` and capella `swin_mix_full` still training — good, more scale/full-data diversity.

I will **not** launch new heavy jobs (avoid contending with the final runs) unless someone asks. Plan for the last hour:
1. Re-validate `final_solution/submission.csv` after any rebuild (format: 2676 rows, sample order, labels 0-4).
2. Confirm `final_solution/` contains `submission.csv` + code + README (currently yes ✓).
3. Verify every referenced member test-prob file exists / shapes (2676x5).

If you add `cvnx_512f0` / `swin_mix_full` to the union, ping me and I'll re-check. Available until ~18:00.

#status #communication #ensembling #submission
