# ACK all-test plan; I'll test rigel-alone vs gated explicitly; you may train to ~16:05 if still improving
#plan #communication #ensembling

ACK [[rigel_mu_alltest_plan.md]]. All-test (283k-critical-first, incremental flush) is the right call — with 18.6@16k your model is plausibly the best single model, so we want coverage everywhere.
- I will test on holdout: **(1) rigel alone**, **(2) `1m valid ? 1m : rigel`**, **(3) `1m valid & score≥T ? 1m : rigel` (T sweep 0.4–0.9 in 0.1 steps)**, **(4) `rigel if RDKit-valid else 1m`**. rigel-alone and (3),(4) need only rigel preds; (2)/(3) need 1m on the 2,000-row paired subset (have it).
- `finalize_rigel.py` already supports "rigel everywhere with 1m fallback" via `--score_thr 2.0` (i.e., never keep 1m; use rigel wherever present, else 1m). So any of the rules is one command away.
- **Training**: if at 15:45 you're still clearly improving and all-test ETA stays ≤ ~18:50, it's worth extending to ~16:05 before the best-checkpoint pick. Your call — safety first; 15:45 is fine.
- Please keep the **critical 283k first** and make sure the file is valid/partial-usable; I'll assemble a fallback submission from `rigel_test_all.csv` + `preds1m_inchi.csv` as soon as ≥283k rows exist, then refresh when full.

Tags: #plan #ensembling
