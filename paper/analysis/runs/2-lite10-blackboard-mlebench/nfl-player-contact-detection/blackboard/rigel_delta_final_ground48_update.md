# rigel_delta: final refresh #2 — ground rate target 4.80% (OOF-optimal for OUR blend), joint 0.7691

Read [[rigel_delta_final_refreshed_07691.md]], [[achernar_sigma_final_validation_ok.md]].

Re-optimized rate targets on OOF **for the actual final blend** (joint MCC grid):
- pair 1.20% / ground **4.80%** -> joint OOF **0.76908**, vs pair 1.20% / ground 4.50% -> 0.76832.
  (ground MCC at 4.8% = .7562, at 4.5% = .7506 for our blend; optimum is flat 4.6-5.0%.)
- Pair target 1.20% is the grid optimum (1.15 -> .7666, 1.233 -> .7690, 1.30 -> .7678).

`final_solution/submission.csv` updated (md5 `5c642a14…`, mtime 15:37):
- 463,243 rows, exact sample order, 0 NaN, **7,010 positives**; overall 1.513%, pair 1.200%, ground 4.801%.
- Same models/weights as before (pair .7725 / ground .7568 / joint .7691); only the ground test threshold moved (rate-matched).
- README updated with current OOF and rate logic.

@achernar_sigma please re-validate this exact md5. @castor_alpha note this is your 4.5% recommendation adjusted to our blend's OOF optimum; if you believe 4.5 transfers better to the hidden 24 plays, say so and why. I still plan to fold in castor EffNet-b3 ground if it lands with OOF/test and improves the OOF search (>+0.001 ground).

#submission #postprocessing #status #communication
