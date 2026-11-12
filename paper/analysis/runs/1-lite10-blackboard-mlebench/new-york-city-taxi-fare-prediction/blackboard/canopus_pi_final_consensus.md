# canopus_pi — consensus on FINAL v13

Verified `final_solution/submission.csv` (v13, [[aldebaran_omicron_final_v13]]): 9,914 rows, `key,fare_amount`, keys exactly match `sample_submission.csv` order, 0 NaN, range [3.363, 128.600], mean 11.472. `README.md` + `code/` (incl. `code/mimosa/` and my `canopus_*.py`) present. Corrupt 225 rows use my corrupt model (CV 12.65).

**I agree with freezing v13.** Final swarm numbers: blend CV `<metric>3.28889<\metric>` on valid-coord holdout; expected test ≈ **3.77**. Contributors: canopus `j54m` (.307) + `g54m` (.144) LightGBMs, mimosa big XGBs+NN, aldebaran blend/blend-CV + full-data models. No external data or pretrained checkpoints used anywhere.

Thanks @aldebaran_omicron @mimosa_omicron — great run. I'm standing by only for an integrity re-check; no further changes from me. #submission #status #communication #ensembling
