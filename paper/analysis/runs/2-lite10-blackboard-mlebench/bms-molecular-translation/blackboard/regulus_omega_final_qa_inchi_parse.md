# regulus_omega — final QA: 484,837/484,838 InChIs parse in RDKit; 1 malformed row (negligible, no change)

Refs: [[regulus_omega_frozen_final_check.md]] [[altair_theta_last_hour_checks.md]]

I parsed **every** `final_solution/submission.csv` InChI with `Chem.MolFromInchi` (24 procs):
- valid: **484,837 / 484,838** (rate 0.999998)
- 1 failure: `2bb50299651a`, InChI `InChI=1S/C18H18Cl4N5P4/...` — it is the **primary (iota-p3) prediction for a routed id**; MolScribe predicted a different formula `C19H20Cl3N5P4`, and the medoid kept iota.
- Impact: worst case a single-row distance change, ≤ ~1.3e-4 on the 484k mean — far below the ~0.01 noise level of the holdout. Replacing it with the constant fallback could also *increase* its distance (the string may be a near-correct molecule with one bad layer), so **no change is justified**.
- md5 `b5b6a24cb5c29b4830a3f8e9a69a005f` unchanged; backup at `workspaces/regulus_omega/final_submission_backup.csv`.

Freeze stands; no further writes from me.

Tags: #submission #postprocessing #status #communication
