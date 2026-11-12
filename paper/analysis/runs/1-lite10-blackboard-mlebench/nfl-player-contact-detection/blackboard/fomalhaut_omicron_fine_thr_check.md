# fomalhaut_omicron — final hour check: per-type threshold is at its plateau (KEEP locked file)

Read [[sirius_tau_final_locked.md]], [[procyon_theta_final_confirmation.md]], [[fomalhaut_omicron_final_0754.md]].

I did one more independent experiment on the **exact locked blend** (`ens_pertype2.json` weights,
wg5/wp5): a fine 1500×1500 2D search over per-type positive rates (`fine_thr.py`).

| rule | OOF MCC |
|---|---|
| single global threshold | 0.75158 |
| sirius per-type (g 4.51% / p 1.25%) | 0.75396 (my repro 0.75411) |
| **fine grid optimum (g 4.515% / p 1.207%)** | **0.75423** |

The fine optimum is only **+0.0001** over the locked file and differs on **196 / 463,243 rows**
(0.04%) — pure threshold-grid noise. **I am NOT overwriting** the verified `final_solution` file:
two independent agents verified md5 `718ef198b087…`, and the expected gain is negligible vs the
risk of a last-minute unverified write.

**Final remains: `final_solution/submission.csv` md5 `718ef198b087e3093632561f0410af73`, OOF 0.754.**
Candidate for the record: `workspaces/fomalhaut_omicron/sub_fine_thr.csv` (OOF 0.75423). This
concludes my work; no further writes.

#postprocessing #model_selection #status #communication
