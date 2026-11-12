# Last probe: rot90/rot270 on the r180-invalid remainder gives NO gain — Rule C is final

Refs: [[altair_epsilon_final_verified.md]] [[alderamin_xi_FINAL_RULEC.md]] [[rigel_mu_freeze.md]]

Checked the only remaining slice (raw rigel invalid **and** rot180 invalid, 1,480 holdout rows; 117,355 test rows in the final rule):
- rot90 becomes RDKit-valid on **0.68%**, rot270 on **0.54%**; when valid they are *worse* than the raw rigel string (rot90 22.2 vs raw 16.8; rot270 12.0 vs raw 9.25).
- Combined mean is flat/slightly worse (19.20 -> 19.21/19.23). Lowering the 1m threshold below 0.78 also hurt (tested 0.6/0.65/0.7).

=> No further rescue is available on the residual slice; **Rule C / `final_solution/submission.csv` (holdout <metric>6.77<\metric>) is the final answer.** Artifact cross-verified identical by three independent builds; frozen.

Tags: #augmentation #insight #postprocessing #status #submission
