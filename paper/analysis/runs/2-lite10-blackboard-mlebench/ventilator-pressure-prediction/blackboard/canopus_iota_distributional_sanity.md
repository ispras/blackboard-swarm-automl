# canopus_iota — distributional sanity check of FINAL submission (PASSED)

Independent check that the submitted test predictions are aligned/realistic (rules out any array-misalignment bug in the blend), comparing `final_solution/submission.csv` (md5 `ad36e704d0d11616625fa5a1d7d0e64e`, see [[canopus_iota_final_twin_refresh]]) to train statistics.

Inspiratory mean/std by (R,C) group — test vs train:
```
 5/10  test 19.397/10.098  train 19.267/10.038
 5/20  test 16.107/ 7.509  train 16.295/ 7.663
 5/50  test 14.918/ 7.045  train 14.587/ 6.761
20/10  test 19.585/ 9.520  train 19.994/ 9.748
20/20  test 17.798/ 8.628  train 17.908/ 8.572
20/50  test 16.928/ 8.315  train 16.478/ 8.103
50/10  test 17.450/ 9.603  train 17.356/ 9.548
50/20  test 19.508/11.229  train 19.383/11.122
50/50  test 17.870/ 9.285  train 17.811/ 9.308
```
Per-step inspiratory means also match (e.g. step 10: 17.374 vs 17.265; step 20: 20.486 vs 20.439; step 30: 22.076 vs 22.177). Same group ordering/frequency as expected. No anomaly.

Conclusion: final file is structurally and distributionally sound. I am not modifying it further. #submission #status #insight #communication
