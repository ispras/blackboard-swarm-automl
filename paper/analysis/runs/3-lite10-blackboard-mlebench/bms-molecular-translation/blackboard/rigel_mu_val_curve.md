# rigel_mu — val curve improving fast: <metric>57.28<\metric> @step4000 (81.2->70.2->64.0->57.3)

Refs: [[rigel_mu_training_resumed.md]] [[alderamin_xi_tta_rot_priority.md]] [[altair_epsilon_tta2_plan.md]]

## Val trajectory (1024 holdout, greedy)
| step | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|
| Lev | 81.16 | 70.24 | 64.04 | **<metric>57.28</metric>** |
Loss 1.16 -> 0.87. Still early (step 4600/20243). This is converging much better than I feared — my model may become competitive on the **both-invalid** rows.

## Reply to alderamin
**Do not pause my training for a 4th TTA proc.** Keep 3 TTA procs + training (the proven config). TTA finishing ~13:35 at 3 procs is fine. My training is now the more valuable lever for the both-invalid rows, and a pause costs ~40 min of convergence.
- Note: if rigel ends up beating TTA on the both-invalid holdout subset, we may simply use **rigel** there and TTA only where rigel is empty. I'll give you `rigel_val20k.csv` at ~15:45; evaluate `both-invalid -> TTA-if-valid-else-rigel` AND `-> rigel always` AND `-> mix by validity` and ship the best.

## Plan
- Cosine ends ~14:50; I'll do a **short warm-restart continuation to ~15:45** (no other changes) if the curve is still improving.
- 15:45–16:15: infer my model on holdout_20k + both-invalid test ids.
- I stay at 3 TTA procs + my training; no 4th proc.

Tags: #status #training #ensembling #communication
