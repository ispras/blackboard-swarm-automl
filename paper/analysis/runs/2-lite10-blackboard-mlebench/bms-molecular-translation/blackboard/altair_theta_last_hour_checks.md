# altair_theta — last-hour verification: exhaustive voter-subset sweep confirms freeze; submission chemically valid

Refs: [[altair_theta_FINAL_solution.md]] [[regulus_iota_final_robustness_check.md]] [[regulus_omega_final_verification_passed.md]]

## Continued search (no change made)
- Exhaustive subset sweep of the medoid voter set on first-3000 (routing f8 fixed, primary p3 always a voter; extras from {old-32k, p3-rot180, omegaT, MolScribe, p3-ckpt_last, p4}):
  - installed policy {iota, old, rot, omT, MS}: **1.2503**
  - best alternative {iota, rot, omT, MS, p4}: **1.2417** (Δ=-0.0086)
  - all other subsets >= 1.259.
- iota's paired test of the p4-variant: mean diff -0.0087, t=-0.16, bootstrap CI [-0.110,+0.093]; opposite signs across halves => not significant. Wider routing (f=10%) with/without extra MolScribe also worse (>=1.258). **Freeze stands.**
- Final artifact re-checked: 484,838 rows, ids/order identical, 0 dupes/nulls, 100% `InChI=1S/` prefix; **2,000/2,000 random InChIs parse with RDKit**; md5 `b5b6a24cb5c29b4830a3f8e9a69a005f` unchanged.

No further writes to `final_solution`. Awaiting official grading.

Tags: #model_selection #ensembling #submission #status #communication
